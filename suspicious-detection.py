import argparse
import logging
import itertools
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import roc_auc_score
from tldextract import TLDExtract

from binlog_reader import binlog_reader

logger = logging.getLogger(__name__)


class DomainFilter(object):
    def __init__(self):
        self._extract = TLDExtract(include_psl_private_domains=True)

    def transform(self, domain):
        return self._extract(domain).registered_domain


def ga_similarities(ipl_1, ipl_2):
    a, b = float(len(ipl_1)), float(len(ipl_2))
    c = float(len(ipl_1 & ipl_2))
    return 0.5 * (c / a + c / b), c / len(ipl_1 | ipl_2), int(c)


def group_activities(all_domains, domain2ip_by_hour):
    logger.debug("Calculating group activity")
    result = {domain: dict() for domain in all_domains}
    pairs = list(filter(lambda (x, y): x < y, itertools.permutations(range(len(domain2ip_by_hour)), 2)))
    feature_name_pattern = "({}, {})_{}"

    for idx, domain in enumerate(all_domains):
        for curr, other in pairs:
            fst_hour, snd_hour = domain2ip_by_hour[curr].get(domain), domain2ip_by_hour[other].get(domain)
            sim, jcd, ln = 0., 0., 0.
            if fst_hour and snd_hour:
                sim, jcd, ln = ga_similarities(fst_hour, snd_hour)

            result[domain].update({feature_name_pattern.format(curr, other, "sim"): sim,
                                   feature_name_pattern.format(curr, other, "jcd"): jcd,
                                   feature_name_pattern.format(curr, other, "length"): ln})
        if (idx + 1) % 100000 == 0:
            logger.debug("Processed %d domains", idx + 1)

    return result


def ranking(ip2d, d2ip, X, y, init_abs_score=10, n_iter=20):
    logger.debug("Calculating rank scores")
    rank_ip = {ip: {'sc_score': 0., 'black_score': 0., 'white_score': 0.} for ip in ip2d}
    rank_d = {d: {'sc_score': 0., 'black_score': 0., 'white_score': 0.} for d in d2ip}

    for dom, cls in zip(X, y):
        if cls == 1:
            rank_d[dom]['sc_score'] = -float(init_abs_score)
            rank_d[dom]['black_score'] = -float(init_abs_score)

        elif cls == -1:
            rank_d[dom]['sc_score'] = float(init_abs_score)
            rank_d[dom]['white_score'] = float(init_abs_score)

    for it in range(n_iter):
        logger.debug("Iteration %d", it + 1)

        for ip in rank_ip:
            rank_ip[ip]['sc_score'] = sum(rank_d[d]['sc_score'] / len(d2ip[d]) for d in ip2d[ip])
            rank_ip[ip]['black_score'] = sum(rank_d[d]['black_score'] / len(d2ip[d]) for d in ip2d[ip])
            rank_ip[ip]['white_score'] = sum(rank_d[d]['white_score'] / len(d2ip[d]) for d in ip2d[ip])

        for domain in rank_d:
            rank_d[domain]['sc_score'] = sum(rank_ip[ip]['sc_score'] / len(ip2d[ip]) for ip in d2ip[domain])
            rank_d[domain]['black_score'] = sum(rank_ip[ip]['black_score'] / len(ip2d[ip]) for ip in d2ip[domain])
            rank_d[domain]['white_score'] = sum(rank_ip[ip]['white_score'] / len(ip2d[ip]) for ip in d2ip[domain])

    return rank_d


def read_logfile(fname, fields=("client_ip", "dname")):
    with open(fname, 'rb') as infile:
        logger.debug("Open file %s", fname)
        reader = binlog_reader(infile, fields)
        return set([tuple([query[fld] for fld in fields]) for query in reader])


def create_indexes(pairs):
    logger.debug("Creating indexes domain2ip & ip2domain")
    domain2ip, ip2domain = dict(), dict()
    for ip, domain in pairs:
        domain2ip.setdefault(domain, set())
        domain2ip[domain].add(ip)
        ip2domain.setdefault(ip, set())
        ip2domain[ip].add(domain)

    return domain2ip, ip2domain


def merge_indexes(indexes):
    logger.debug("Merge %d indexes to one", len(indexes))
    ip2domain_full, domain2ip_full = dict(), dict()
    for (d2ip, ip2d) in indexes:
        for domain in d2ip:
            domain2ip_full.setdefault(domain, set())
            domain2ip_full[domain] |= d2ip[domain]

        for ip in ip2d:
            ip2domain_full.setdefault(ip, set())
            ip2domain_full[ip] |= ip2d[ip]
    return domain2ip_full, ip2domain_full


def create_domain_indexes(hosts):
    logger.debug("Creating indexes host2domain & domain2host")
    df = DomainFilter()
    domain2host, host2domain = dict(), dict()
    for host in hosts:
        domain = df.transform(host)
        domain2host.setdefault(domain, set())
        domain2host[domain].add(host)
        host2domain[host] = domain

    return domain2host, host2domain


def prepare_trainset(blacklist, whitelist, all_domains):
    df = DomainFilter()
    with open(blacklist, 'r') as infile:
        blacklist_domains = [df.transform(line.strip()) for line in infile]
    with open(whitelist, 'r') as infile:
        whitelist_domains = [df.transform(line.strip()) for line in infile]

    pos, neg = set(filter(lambda d: d, blacklist_domains)), set(filter(lambda d: d, whitelist_domains))

    inter = pos & neg
    logger.debug("Positive size %d, Negative size %d, Intersection %d, All domains %d",
                 len(pos), len(neg), len(inter), len(all_domains))
    pos, neg = (pos - inter) & all_domains, (neg - inter) & all_domains
    logger.debug("Positive after %d, Negative after %d", len(pos), len(neg))

    X = list(pos) + list(neg)
    y = [1 for _ in range(len(pos))] + [-1 for _ in range(len(neg))]
    return np.array(X), np.array(y)


def join_features_by_keys(keys, features_list):
    logger.debug("Joining features")
    tmp_full = {key: dict() for key in keys}
    for domain in tmp_full:
        for features in features_list:
            tmp_full[domain].update(features[domain])

    all_features = sorted(tmp_full[keys[0]].keys())
    logger.debug("Features - %s", ','.join(all_features))
    return np.array([[tmp_full[domain][f] for f in all_features] for domain in keys])


def grid_search(scoring, X_train, y_train, X_test, y_test):
    for classifier in scoring:
        logger.debug("Griding clf %s", classifier.__name__)
        for idx, param in enumerate(scoring[classifier]):
            clf = classifier(**dict(param))
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scoring[classifier][param].append(roc_auc_score(y_test, y_pred))
            if (idx + 1) % 10 == 0:
                logger.debug("Iteration #%d of %d", idx + 1, len(scoring[classifier]))


def make_predictor(X, y, ip2domain_full, domain2ip_full, const_features, n_folds=5):
    logger.debug("Creating classifier, n_folds = %d", n_folds)
    skf = StratifiedKFold(y, n_folds=n_folds)

    clfs = [
        (AdaBoostClassifier, {"n_estimators": [30, 50, 70, 100, 120, 150],
                              "learning_rate": [1., 0.8, 0.5, 0.1, 0.05]}),
        (RandomForestClassifier, {"n_estimators": range(10, 101, 10),
                                  "max_features": ["sqrt", "log2", None]}),
        (GradientBoostingClassifier, {"loss": ['deviance', 'exponential'],
                                      "learning_rate": [0.01, 0.05, 0.1, 0.5],
                                      "n_estimators": [50, 100, 150, 200],
                                      "max_depth": range(2, 9)})
    ]
    scoring = {clf: {frozenset(comb.items()): [] for comb in ParameterGrid(params)}
               for (clf, params) in clfs}

    for idx, (train_index, test_index) in enumerate(skf):
        logger.debug("Folding iteration #%d", idx + 1)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rank_features = ranking(ip2domain_full, domain2ip_full, X_train, y_train,
                                init_abs_score=10, n_iter=20)

        X_features_train = join_features_by_keys(X_train, [const_features, rank_features])
        X_features_test = join_features_by_keys(X_test, [const_features, rank_features])

        grid_search(scoring, X_features_train, y_train, X_features_test, y_test)

    for clf in scoring:
        for param in scoring[clf]:
            scoring[clf][param]['mean'] = np.mean(scoring[clf][param])
            scoring[clf][param]['median'] = np.median(scoring[clf][param])
    from pprint import pprint
    pprint(scoring)
    return scoring


def processing(logfiles, blacklist, whitelist):
    queries = [read_logfile(fn, ("client_ip", "dname")) for fn in sorted(logfiles)]
    hosts = set([domain for (ip, domain) in itertools.chain.from_iterable(queries)])
    domain2host, host2domain = create_domain_indexes(hosts)

    queries = [[(ip, host2domain[domain]) for (ip, domain) in hour] for hour in queries]
    queries = [set(filter(lambda (_, d): d, hour)) for hour in queries]

    small_indexes = [create_indexes(hour) for hour in queries]
    domain2ip_full, ip2domain_full = merge_indexes(small_indexes)
    all_domains = set(domain2ip_full.keys())
    domain2ip_by_hour = [d2ip for (d2ip, ip2d) in small_indexes]

    X, y = prepare_trainset(blacklist, whitelist, all_domains)
    ga_features = group_activities(all_domains, domain2ip_by_hour)

    clf = make_predictor(X, y, ip2domain_full, domain2ip_full, ga_features, n_folds=4)
    # y_pred = clf.predict(X)


def main():
    parser = argparse.ArgumentParser(description="Suspicious domain detector (used querylog)")

    parser.add_argument('-f', '--files', help='Files with logs', required=True, nargs='*')
    parser.add_argument('-b', '--blacklist', help='Path to blacklist', required=True, type=str)
    parser.add_argument('-w', '--whitelist', help='Path to whitelist', required=True, type=str)
    parser.add_argument('-v', '--verbose', help='Verbose flag', action='store_const', dest="loglevel",
                        const=logging.DEBUG, default=logging.WARNING)

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=args.loglevel)
    return processing(args.files, args.blacklist, args.whitelist)


if __name__ == '__main__':
    exit(main())
