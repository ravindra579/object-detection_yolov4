from keras.models import Model
import struct
from keras.models import load_model
from yolov4_model.py import yolov4_model
model=yolov4_model()
class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,    = struct.unpack('i', w_f.read(4))
            minor,    = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))
            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                print("reading 64 bytes")
                w_f.read(8)
            else:
                print("reading 32 bytes")
                w_f.read(4)
            transpose = (major > 1000) or (minor > 1000)
            binary = w_f.read()
        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    def load_weights(self, model):
        count = 0
        ncount = 0
        for i in range(161):
            try:
                conv_layer = model.get_layer('convn_' + str(i))
                filter = conv_layer.kernel.shape[-1]
                nweights = np.prod(conv_layer.kernel.shape) 
                print("loading weights of convolution #" + str(i)+ "- nb parameters: "+str(nweights+filter))
                if i  in [138, 149, 160]:
                    print("Special processing for layer "+ str(i))
                    bias  = self.read_bytes(filter)
                    weights = self.read_bytes(nweights) 
                else:                    
                    bias  = self.read_bytes(filter) 
                    scale = self.read_bytes(filter) 
                    mean  = self.read_bytes(filter) 
                    var   = self.read_bytes(filter) 
                    weights = self.read_bytes(nweights) 
                    bias = bias - scale  * mean / (np.sqrt(var + 0.00001)) 
                    weights = np.reshape(weights,(filter,int(nweights/filter))) 
                    A = scale / (np.sqrt(var + 0.00001))
                    A= np.expand_dims(A,axis=0)
                    weights = weights* A.T
                    weights = np.reshape(weights,(nweights))
                weights = weights.reshape(list(reversed(conv_layer.get_weights()[0].shape)))                 
                weights = weights.transpose([2,3,1,0])
                if len(conv_layer.get_weights()) > 1:
                    a=conv_layer.set_weights([weights, bias])
                else:    
                    a=conv_layer.set_weights([weights])
                count = count+1
                ncount = ncount+nweights+filter
            except ValueError:
                print("no convolution #" + str(i)) 
        print(count, "Conv normalized layers loaded ", ncount, " parameters")
    def reset(self):
        self.offset = 0
weight_reader=WeightReader("yolov4.weights")
weight_reader.load_weights(model)
model.save_weights("check.h5")
model.load_weights("check.h5")
