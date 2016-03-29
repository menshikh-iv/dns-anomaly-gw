from struct import unpack as u
from functools import partial


def query_factory(fields):
    query = {
            "version": partial(lambda packet: u("B", packet[:1])[0]),
            "rcode_and_flags": partial(lambda packet: u("B", packet[1: 2])[0]),
            "qtype": partial(lambda packet: u("H", packet[2: 4])[0]),
            "timestam_usec": partial(lambda packet: u("Q", packet[6: 14])[0]),
            "reserved2": partial(lambda packet: u("I", packet[14: 18])[0]),
            "reserved3": partial(lambda packet: u("I", packet[18: 22])[0]),
            "reserved4": partial(lambda packet: u("I", packet[22: 26])[0]),
            "client_ip": partial(lambda packet: ".".join(str(x) for x in u("BBBB", packet[26: 30]))),
            "profile_id": partial(lambda packet: u("I", packet[30: 34])[0]),
            "latency_usec": partial(lambda packet: u("I", packet[34: 38])[0]),
            "cats": partial(lambda packet: filter(lambda x: x != 0, u("BBBBBBBB", packet[38: 46]))),
            "reserved5": partial(lambda packet: u("I", packet[46: 50])[0]),
            "reserved6": partial(lambda packet: u("I", packet[50: 54])[0]),
            "dname": partial(lambda packet: "".join(u("c" * len(packet[54:]), packet[54:]))),
    }
    new_query = {key: func for key, func in query.items() if key in fields}
    return lambda length, packet: {key: func(packet) for key, func in new_query.items()}


def binlog_reader(fd, fields, buffer_size=2**20):
    get_query = query_factory(fields)
    buff = fd.read(max(2, buffer_size))

    while buff:
        length = u("H", buff[: 2])[0]
        if len(buff) <= length:
            buff += fd.read(buffer_size - len(buff))
        packet = buff[2: length]
        yield get_query(length, packet)
        buff = buff[length:]
        if not buff or len(buff) == 1:
            buff += fd.read(buffer_size - len(buff))
