import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def f(x):
    import os
    # global _worker_num
    p = multiprocessing.current_process()
    print(p, type(p), p.name, p._identity[0], type(p._identity[0]), p.ident)
    return x * x


def init_worker(lock, array_pid):
    print(array_pid, len(array_pid))
    child_process = multiprocessing.current_process()

    lock.acquire()
    num_worker = None
    for i in range(len(array_pid)):
        print(array_pid[i])
        if array_pid[i] == -1:
            num_worker = i
            array_pid[i] = child_process.ident
            break
    print(num_worker, child_process.ident)
    lock.release()

num_worker = 6
lock = multiprocessing.Lock()
array_pid = multiprocessing.Array('i', num_worker)
for i in range(num_worker):
    array_pid[i] = -1


# with ProcessPoolExecutor(
#     max_workers=4,
# ) as executor:
#     print(executor._processes)
#     results = executor.map(f, range(6))

with ProcessPoolExecutor(
    max_workers=4,
    initializer=init_worker,
    initargs=(lock, array_pid)
) as executor:
    print(executor._processes)
    results = executor.map(f, range(6))

exit()
# global _worker_num
# def set_worker_index(i):
#     global _worker_num
#     _worker_num = i

p = multiprocessing.Pool(processes=3)
# children = multiprocessing.active_children()
# for i, child in enumerate(children):
#     child.submit(set_worker_index)

# print(children)
print(p.map(f, range(6)))
p.close()
p.join()

p = multiprocessing.Pool(processes=3)
print(p.map(f, range(6)))
print(p.map(f, range(6)))

p.close()
p.join()


# print(multiprocessing.current_process())
# p = multiprocessing.current_process()
# print(p._identity)
