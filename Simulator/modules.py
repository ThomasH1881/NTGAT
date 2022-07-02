import numpy as np
import queue
import collections

class Decoder_Unit:

    def __init__(self, ID, delay):
        self.ID = ID
        self.idle = True  #idle at the beginning
        self.delay = delay    # the delay cycles of sending a request from DU to FIFO
        self.time_stamp = 0    #time
        self.nlist = []        #list of edges
        self.node_ID = 0       #node ID

    def register(self, node_ID, nlist, time):
        self.node_ID = node_ID       # register a new node and start processing
        self.idle = False            # no longer idle
        self.nlist = nlist           # the list of all its neighbours
        self.time_stamp = time + 1   # register takes one cycle

    def step(self, fifo):
        n = self.nlist.pop()   # get one neighbour from list
        chunk_ID = n % 256  # compute the ID of chunk this node belongs
        if (fifo[chunk_ID].full()==False):
            fifo[chunk_ID].put(((n, self.ID, self.time_stamp + self.delay)))   # fifo is not full，send to fifo
        else:
            self.nlist.append(n)   # fifo is full, put back to the list
        if (self.nlist == []):       # empty, become idle
            self.idle = True
        self.time_stamp = self.time_stamp + 1 # step takes one cycle

class e_FIFO:

    def __init__(self, ID, Size):
        self.ID = ID
        self.Q = queue.Queue(maxsize=Size)   # fifo queue
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
        x = self.Q.get()              # get an event (node_ID, SSR_ID, time_s)
        SSR[x[1]].count += 1          # SSR counts another node
        SSR[x[1]].regs.append(x[0])      # node_ID get in queue
        SSR[x[1]].time_stamp = x[2] + 2 + 11   #SSR time_stamp updates, sending and pipeline cycles.
        self.time_stamp += 1

class Swap_Shift_Register:       # the sorting processor was originally called SSR, this simulator continues to use this name.

    def __init__(self, ID):
        self.ID = ID
        self.idle = True
        self.time_stamp = 0
        self.count = 0
        self.regs = []         # regs preserving all the nodes stored

    def register(self, node_ID, degree=0, depth=0):         # register with node ID and node degree
        self.node_ID = node_ID
        self.regs = []
        self.degree = degree
        self.depth = depth      # depth is the number of total neighbours to process for this node.
        self.idle = False

    def step(self, VP, total):
        # one node finishes, print a message
        print("node",self.node_ID, "in", "SSR", self.ID,"finishes, copying to VP", self.ID, "at time:", self.time_stamp)
        # start to register for VP.
        VP[self.ID].regs = self.regs    # simply copy all the nodes
        VP[self.ID].idle = False        # start working
        VP[self.ID].ID = self.ID
        # copy other parameters
        VP[self.ID].node_ID = self.node_ID
        VP[self.ID].time_stamp = self.time_stamp + 1
        VP[self.ID].depth = self.depth
        # global statistics.
        total[0] += self.depth
        VP[self.ID].finished = 0
        self.idle = True
        self.regs = []
        self.count = 0
        self.time_stamp += 1

class Vector_Processor:

    def __init__(self, ID, num_cache = 256, delay = 0, time = 0):
        self.ID = ID
        self.idle = True
        self.delay = delay     # cycles of transferring data from Cache
        self.time = time       # cycles for MAC

        self.num_cache = num_cache
        self.semaphore = 5         # can buffer 5 features at the same time
        self.time_stamp = 0
        self.regs = []
        self.depth = 0
        self.finished = 0
        self.node_ID = 0
        self.MAC_working = False   # if is computing MAC
        self.MAC_finish_time = 0   # next time when MAC finishes
        self.buffered_vectors = 0  # number of buffered vectors

    def step(self, CB, hmf):
        if (self.finished == self.depth):
            self.idle = True
            # VP finishes, print a message
            print("node", self.node_ID, "in", "VP", self.ID, "finishes at time:", self.time_stamp)
            #print("hit:", hmf[0], "miss:", hmf[1])
            self.time_stamp += 1
        else:
            flag = 0
            if (self.MAC_working == True and self.time_stamp == self.MAC_finish_time):  # MAC working and finishes
                self.MAC_working = False
                self.buffered_vectors -= 1
                self.semaphore += 1
                self.finished += 1

            if (self.semaphore > 0 and self.regs != []):
                x = self.regs.pop()    # fetch features of node_x
                cache_ID = x % self.num_cache # compute which cache node_x is in
                if (CB[cache_ID].fifo.full()):
                    self.regs.append(x)     # full, put back to regs
                else:
                    self.semaphore -= 1
                    CB[cache_ID].fifo.put((x, self.ID))  # ((node_ID, VP_ID, time))   # put an event to Cache
                    CB[cache_ID].fifo_time.append(self.time_stamp + self.delay)       # mark the time of event

            if (self.MAC_working == False and self.buffered_vectors > 0):   # not working and has buffered vector
                self.MAC_working = True
                self.MAC_finish_time = self.time_stamp + self.time
                flag = 1

            if (flag == 1):       # next time_stamp should be : MAC time, if MAC working; 1, elsewise.
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
        if (self.AXI_return):       # AXI returns data to cache
            VP[self.AXI_VP_ID].buffered_vectors += 1
            for i in self.DDRdic[self.AXI_node_ID]:    # send back to all VPs in dictionary waiting for this feature
                VP[i].buffered_vectors += 1
            del[self.DDRdic[self.AXI_node_ID]]
            index = self.DDRlist.index(self.AXI_node_ID)  # clear the request in DDR list
            self.DDRlist.pop(index)
            self.cache.replace(self.AXI_node_ID)     # cache replacement
            self.AXI_return = False
            self.time_stamp = self.time_stamp + self.fetch_time   # wait for fetch_time cycles
        elif (self.fetch_start):                  # start fetching
            VP[self.VP_ID].buffered_vectors += 1
            self.fetch_start = False
            self.time_stamp = self.time_stamp + self.fetch_time
        else:
            if (self.time_stamp < self.fifo_time[-1]):       # update to fifo time
                self.time_stamp = self.fifo_time[-1]
            else:
                self.fifo_time.pop()
                (self.node_ID, self.VP_ID) = self.fifo.get()   # get event
                if (self.cache.fetch(self.node_ID, hmf)):
                    self.fetch_start = True                        # feature in cache, simply send data
                    self.time_stamp = self.time_stamp + self.delay
                else:
                    if self.node_ID in self.DDRlist:                # not in cache:
                        self.DDRdic[self.node_ID].append(self.VP_ID)  # if DDR list? add to dictionary and just wait.
                        self.time_stamp = self.time_stamp + 1
                    else:
                        self.DDRlist.append(self.node_ID)       # not in DDR list. send request to DDR and create a DDR dictionary.
                        self.DDRdic[self.node_ID]=[]
                        DDR[self.node_ID % 4].waitlist.put((self.node_ID, self.ID, (self.time_stamp + self.delay), self.VP_ID))  #delay?
                        hmf[2] += 1
                        self.time_stamp = self.time_stamp + 1
                    #AXI[int(node_ID % 4)].


class LRUCache:  # an LRU cache. hmf counts hit, miss, fetch

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
            tmp = self.cache.pop(0)
            self.cache.append(x)


class DDRModel:  # Alveo U200 have 4 DDRs, each has 5 channels

    def __init__(self, ID, length, delay, fetch_time):
        self.time_stamp = 0
        self.ID = ID
        self.length = length
        self.delay = delay
        self.fetch_time = fetch_time
        self.timelist = []
        self.waitlist = queue.Queue()  # ((node_ID, CB_ID, time, VP_ID))

        # waitlist stores all the requests. timelist stores current 5 requests under processing

    def step(self, CB):
        if (len(self.timelist) < self.length and not self.waitlist.empty()):
            x = self.waitlist.get()
            self.timelist.append((x[0], x[1], (self.time_stamp + self.fetch_time), x[3]))
        for i in range(len(self.timelist)):
            if (self.time_stamp >= self.timelist[i][2]):             # time_stamp larger than event time
                if (CB[self.timelist[i][1]].AXI_return == False):    # CB is not working, can transfer now.
                    self.time_stamp += self.delay
                    x = self.timelist.pop(i)
                    CB[x[1]].AXI_return = True                       # results returned
                    CB[x[1]].AXI_node_ID = x[0]
                    CB[x[1]].AXI_VP_ID = x[3]
                    return
        if (self.time_stamp < self.timelist[0][2]):      # next time should be either next event time, or one cycle if dealing with request
            self.time_stamp = self.timelist[0][2]
        else:
            self.time_stamp = self.time_stamp + 1

'''
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
'''

