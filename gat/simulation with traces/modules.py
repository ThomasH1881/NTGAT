import numpy as np
import queue
import collections

class Decoder_Unit:

    def __init__(self, ID, delay):
        self.ID = ID
        self.idle = True  #idle at the beginning
        self.delay = delay    #
        self.time_stamp = 0    #time
        self.nlist = []        #list of edges
        self.node_ID = 0       #node ID

    def register(self, node_ID, nlist, time):
        self.node_ID = node_ID       #绑定一个新节点开始处理
        self.idle = False            #不再空闲
        self.nlist = nlist           #所有邻点的列表
        self.time_stamp = time + 1

    def step(self, fifo):
        n = self.nlist.pop()
        chunk_ID = n % 256  # 计算chunk的ID
        if (fifo[chunk_ID].full()==False):
            fifo[chunk_ID].put(((n, self.ID, self.time_stamp + self.delay)))   #fifo未满，移入fifo
        else:
            self.nlist.append(n)   #fifo已满，队列不动
        if (self.nlist == []):       #empty,idle
            self.idle = True
        self.time_stamp = self.time_stamp + 1

class e_FIFO:

    def __init__(self, ID, Size):
        self.ID = ID
        self.Q = queue.Queue(maxsize=Size)
        self.time_stamp = 0

    def full(self):
        return self.Q.full()
    def empty(self):
        return self.Q.empty()
    def get(self):
        return self.Q.get()
    def put(self, x):
        self.Q.put(x)

    def step(self, SSR):
        x = self.Q.get()              #取出事件(node_ID, SSR_ID, time_s)
        SSR[x[1]].count += 1
        SSR[x[1]].regs.append(x[0])      #node_ID加入队列
        SSR[x[1]].time_stamp = x[2] + 2 + 11   #SSR time_stamp更新
        self.time_stamp += 1

class Swap_Shift_Register:

    def __init__(self, ID):
        self.ID = ID
        self.idle = True
        self.time_stamp = 0
        self.count = 0
        self.regs = []

    def register(self, node_ID, degree=0):         #用节点ID和度数注册
        self.node_ID = node_ID
        self.regs = []
        self.degree = degree
        self.idle = False

    def step(self, VP, trunc, total):
        print("node",self.node_ID, "in", "SSR", self.ID,"finishes, copying to VP", self.ID, "at time:", self.time_stamp)
        VP[self.ID].regs = [self.node_ID] + trunc[self.node_ID]
        VP[self.ID].idle = False
        VP[self.ID].ID = self.ID
        VP[self.ID].node_ID = self.node_ID
        VP[self.ID].time_stamp = self.time_stamp + 1
        VP[self.ID].depth = len(trunc[self.node_ID]) + 1
        total[0] += len(trunc[self.node_ID])
        VP[self.ID].finished = 0
        self.idle = True
        self.regs = []
        self.count = 0
        self.time_stamp += 1

class Vector_Processor:

    def __init__(self, ID, num_cache = 256, delay = 0, time = 0):
        self.ID = ID
        self.idle = True
        self.delay = delay
        self.time = time

        self.num_cache = num_cache
        self.semaphore = 5
        self.time_stamp = 0
        self.regs = []
        self.depth = 0
        self.finished = 0
        self.node_ID = 0
        self.MAC_working = False
        self.MAC_finish_time = 0
        self.buffered_vectors = 0

    def step(self, CB, hm):
        if (self.finished == self.depth):
            self.idle = True
            print("node", self.node_ID, "in", "VP", self.ID, "finishes at time:", self.time_stamp)
            #print("hit:", hm[0], "miss:", hm[1])
            self.time_stamp += 1
        else:
            flag = 0
            if (self.MAC_working == True and self.time_stamp == self.MAC_finish_time):
                self.MAC_working = False
                self.buffered_vectors -= 1
                self.semaphore += 1
                self.finished += 1

            if (self.semaphore > 0 and self.regs != []):
                x = self.regs.pop()    #访存node_x的features
                cache_ID = x % self.num_cache
                if (CB[cache_ID].fifo.full()):
                    self.regs.append(x)
                else:
                    self.semaphore -= 1
                    CB[cache_ID].fifo.put((x, self.ID))  # ((node_ID, VP_ID, time))
                    CB[cache_ID].fifo_time.append(self.time_stamp+self.delay)

            if (self.MAC_working == False and self.buffered_vectors > 0):
                self.MAC_working = True
                self.MAC_finish_time = self.time_stamp + self.time
                flag = 1

            if (flag == 1):
                self.time_stamp = self.time_stamp + self.time
            else:
                self.time_stamp += 1


class CacheBlock:

    def __init__(self, ID, maxsize, fifosize, delay, fetch_time):
        self.cache = LRUCache(maxsize)
        self.ID = ID
        self.fifo = queue.Queue(maxsize=fifosize)   # ((node_ID, VP_ID))
        self.fifo_time = []
        self.delay = delay
        self.fetch_time = fetch_time
        self.time_stamp = 0
        self.node_ID = 0
        self.VP_ID = 0
        self.req_finish = 0
        self.fetch_start = False
        self.AXI_start = 0
        self.AXI_return = False
        self.AXI_node_ID = 0
        self.AXI_VP_ID = 0
        self.DDRlist = []
        self.DDRdic = {}

    def step(self, VP, DDR, hmf):
        if (self.AXI_return):
            VP[self.AXI_VP_ID].buffered_vectors += 1
            for i in self.DDRdic[self.AXI_node_ID]:
                VP[i].buffered_vectors += 1
            del[self.DDRdic[self.AXI_node_ID]]
            index = self.DDRlist.index(self.AXI_node_ID)
            self.DDRlist.pop(index)
            self.cache.replace(self.AXI_node_ID)
            self.AXI_return = False
            self.time_stamp = self.time_stamp + self.fetch_time
        elif (self.fetch_start):
            VP[self.VP_ID].buffered_vectors += 1
            self.fetch_start = False
            self.time_stamp = self.time_stamp + self.fetch_time
        else:
            if (self.time_stamp < self.fifo_time[-1]):
                self.time_stamp = self.fifo_time[-1]
            else:
                self.fifo_time.pop()
                (self.node_ID, self.VP_ID) = self.fifo.get()
                if (self.cache.fetch(self.node_ID, hmf)):
                    self.fetch_start = True
                    self.time_stamp = self.time_stamp + self.delay
                else:
                    if self.node_ID in self.DDRlist:
                        self.DDRdic[self.node_ID].append(self.VP_ID)
                        self.time_stamp = self.time_stamp + 1
                    else:
                        self.DDRlist.append(self.node_ID)
                        self.DDRdic[self.node_ID]=[]
                        DDR[self.node_ID % 4].waitlist.put((self.node_ID, self.ID, (self.time_stamp + self.delay), self.VP_ID))  #delay?
                        hmf[2] += 1
                        self.time_stamp = self.time_stamp + 1
                        #AXI[int(node_ID % 4)].


class LRUCache:

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.size = 0
        self.cache = []
        self.hit = 0
        self.miss = 0

    def fetch(self, x, hmf):
        if x in self.cache:
            index = self.cache.index(x)
            self.cache.pop(index)
            self.cache.append(x)
            self.hit += 1
            hmf[0] += 1
            return True
        else:
            self.miss += 1
            hmf[1] += 1
            return False

    def replace(self, x):
        if self.size < self.maxsize:
            self.cache.append(x)
            self.size += 1
        else:
            self.cache.pop(0)
            self.cache.append(x)


class DDRModel:

    def __init__(self, ID, length, delay, fetch_time):
        self.time_stamp = 0
        self.ID = ID
        self.length = length
        self.delay = delay
        self.fetch_time = fetch_time
        self.timelist = []
        self.waitlist = queue.Queue()  # ((node_ID, CB_ID, time, VP_ID))

    def step(self, CB):
        if (len(self.timelist) < self.length and not self.waitlist.empty()):
            x = self.waitlist.get()
            self.timelist.append((x[0], x[1], (self.time_stamp + self.fetch_time), x[3]))
        for i in range(len(self.timelist)):
            if (self.time_stamp >= self.timelist[i][2]):
                if (CB[self.timelist[i][1]].AXI_return == False):
                    self.time_stamp += self.delay
                    x = self.timelist.pop(i)
                    CB[x[1]].AXI_return = True
                    CB[x[1]].AXI_node_ID = x[0]
                    CB[x[1]].AXI_VP_ID = x[3]
                    return
        if (self.time_stamp < self.timelist[0][2]):
            self.time_stamp = self.timelist[0][2]
        else:
            self.time_stamp = self.time_stamp + 1

class Multiplier:

    def __init__(self, DSP):
        self.DSP = DSP

    def matrix_multipy(self, num_nodes, input_size, output_size):

        return

    def vector_multipy(self, input_size, output_size):

        return

    def attention_multipy(self, input_size, outputs_size, heads = 0):

        return

    #def


