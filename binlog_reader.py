#from struct import unpack as u
from struct import unpack_from as u
from functools import partial


def query_factory(fields):
    query = {
            "rcode": partial(lambda packet, offset, length: u("B", packet, offset + 3)[0]),
            "qtype": partial(lambda packet, offset, length: u("H", packet, offset + 4)[0]),
            "timestam_usec": partial(lambda packet, offset, length: u("Q", packet, offset + 8)[0]),
            "client_ip": partial(lambda packet, offset, length:".".join(str(x) for x in u("BBBB", packet, offset + 28))),
            "profile_id": partial(lambda packet, offset, length: u("I", packet, offset + 32)[0]),
            "latency_usec": partial(lambda packet, offset, length: u("I", packet, offset + 36)[0]),
            "cats": partial(lambda packet, offset, length: filter(lambda x: x != 0, u("BBBBBBBB", packet, offset + 40))),
            "reserved5": partial(lambda packet, offset, length: u("I", packet, offset + 48)[0]),
            "reserved6": partial(lambda packet, offset, length: u("I", packet, offset + 52)[0]),
            "dname": partial(lambda packet, offset, length: "".join(u("c" * (length - 56), packet, offset + 56)))
    }
    if fields == ["*"]:
        fields = query.keys()
    new_query = {key: func for key, func in query.items() if key in fields}
    return lambda offset, packet, length: {key: func(packet, offset, length) for key, func in new_query.iteritems()}

            
def binlog_reader(fd, fields):
    get_query = query_factory(fields)
    buff = fd.read(-1)
    offset = 0
    
    while offset < len(buff):
        length, = u("H", buff, offset)
        yield get_query(offset, buff, length)
        offset += length
