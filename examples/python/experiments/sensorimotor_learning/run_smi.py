import random
from simple_image_dataset import load_simple_images
from brainblocks.blocks import BlankBlock, PatternPooler, ContextLearner, PatternClassifier, PatternClassifierDynamic
from brainblocks.tools import HyperGridTransform


def get_bits_from_hg(tensor):
    bits = []
    i = 0
    for grid in tensor[0]:
        for bit in grid:
            if bit == True:
                bits.append(1)
            else:
                bits.append(0)
            i += 1
    return bits


def get_acts_from_hg(tensor):
    acts = []
    i = 0
    for grid in tensor[0]:
        for bit in grid:
            if bit == True:
                acts.append(i)
            i += 1
    return acts

# ==============================================================================
# SimpleShapesEnvironment
# ==============================================================================
class SimpleShapesEnvironment:
    def __init__(self):
        self.data = load_simple_images()
        self.image_size = (32, 32)
        self.sensor_size = (8, 8)
        self.sensor_bounds = (24, 24)
        
        self.image_num_bits = self.image_size[0] * self.image_size[1]
        self.sensor_num_bits = self.sensor_size[0] * self.sensor_size[1]
        
        self.image = [0 for _ in range(self.image_num_bits)]

    def __in_bounds(self, location=(0, 0)):
        if location[0] < 0 or location[0] > self.sensor_bounds[0] - 1:
            print('Warning: location[x] must be between 0 and 23 inclusive')
            return False
        if location[1] < 0 or location[1] > self.sensor_bounds[1] - 1:
            print('Warning: location[y] must be between 0 and 23 inclusive')
            return False
        return True

    def set_image(self, image_bits=[]):
        self.image = image_bits

    def get_random_location(self):
        x = random.randint(0, self.sensor_bounds[0] - 1)
        y = random.randint(0, self.sensor_bounds[1] - 1)
        return (x, y)

    def get_feature(self, location=(-1, -1)):
        feature = [0 for _ in range(self.sensor_num_bits)]

        # check if location is in bounds
        if not self.__in_bounds(location):
            return feature

        # get sensory feature bits
        j = 0
        for y in range(location[1], location[1] + self.sensor_size[1]):
            for x in range(location[0], location[0] + self.sensor_size[0]):
                i = (y * self.image_size[0]) + x
                feature[j] = self.image[i]
                j += 1

        return feature

    def get_feature_location(self):
        feature = [0 for _ in range(self.sensor_num_bits)]
        location = (-1, -1)

        for i in range(100):
            location = self.get_random_location()
            feature = self.get_feature(location)

            # get count of bits in middle of feature
            count = 0
            for y in range(2, 6):
                for x in range(2, 6):
                    i = (y * self.sensor_size[0]) + x
                    if feature[i] == 1:
                        count += 1

            # if enough bits in middle of feature then return
            if count > 12:
                break

        return (feature, location)

    def print_img(self, location=(-1, -1)):
        print_array = ['-' for _ in range(self.image_num_bits)]

        # if no location supplied then dont print feature
        print_feature_flag = True
        if location == (-1, -1):
            print_feature_flag = False

        # check if location is in bounds
        elif not self.__in_bounds(location):
            return

        # set active bits from image
        for i in range(self.image_num_bits):
            if self.image[i] == 1:
                print_array[i] = '+'

        # set active bits from location's feature
        if print_feature_flag:
            j = 0
            for y in range(location[1], location[1] + self.sensor_size[1]):
                for x in range(location[0], location[0] + self.sensor_size[0]):
                    i = (y * self.image_size[0]) + x
                    if self.image[i] == 0:
                        print_array[i] = '0'
                    else:
                        print_array[i] = '1'
                    j += 1

        # print array
        for y in range(self.image_size[1]):
            for x in range(self.image_size[0]):
                i = (y * self.image_size[0]) + x
                print(print_array[i], end=' ')
            print()

# ==============================================================================
# SensoriMotorInference
# ==============================================================================
class SensoriMotorInference():
    def __init__(self):

        self.window = 10

        self.actarrays = [[0 for _ in range(8)] for _ in range(self.window)]

        self.hg = HyperGridTransform(
            num_grids=8,
            num_bins=8,
            num_input_dims=2,
            min_period=0.0,
            max_period=64.0)

        self.bbl = BlankBlock(num_s=64) # for sensory location
        self.bbf = BlankBlock(num_s=64) # for sensory features

        self.pp = PatternPooler(
            num_s=64,      # number of statelets
            num_as=8,      # number of active statelets
            perm_thr=20,   # receptor permanence threshold
            perm_inc=2,    # receptor permanence increment
            perm_dec=1,    # receptor permanence decrement
            pct_pool=0.3,  # pooling percentage
            pct_conn=1.0,  # initially connected percentage
            pct_learn=1.0) # learn percentage

        self.cl = ContextLearner(
            num_spc=10,  # number of statelets per column
            num_dps=10,  # number of coincidence detectors per statelet
            num_rpd=8,   # number of receptors per coincidence detector
            d_thresh=5,  # coincidence detector threshold
            perm_thr=20, # receptor permanence threshold
            perm_inc=2,  # receptor permanence increment
            perm_dec=1)  # receptor permanence decrement

        self.bbpc = BlankBlock(num_s=640) # for pattern classifier

        
        self.pc = PatternClassifier(
            num_l=9,       # number of labels
            num_s=512,     # number of statelets
            num_as=8,      # number of active statelets
            perm_thr=20,   # receptor permanence threshold
            perm_inc=2,    # receptor permanence increment
            perm_dec=1,    # receptor permanence decrement
            pct_pool=0.8,  # pooling percentage
            pct_conn=0.5,  # initially connected percentage
            pct_learn=0.3) # learn percentage
        
        '''
        self.pc = PatternClassifierDynamic(
            num_s=512,     # number of statelets
            num_as=8,      # number of active statelets
            num_spl=32,    # number of statelets per label
            perm_thr=20,   # receptor permanence threshold
            perm_inc=2,    # receptor permanence increment
            perm_dec=1,    # receptor permanence decrement
            pct_pool=0.8,  # pooling percentage
            pct_conn=0.5,  # initially connected percentage
            pct_learn=0.3) # learn percentage
        '''

        # connect feature blank block to pattern pooler
        self.pp.input.add_child(self.bbf.output)

        # connect pattern pooler output to context learner input
        self.cl.input.add_child(self.pp.output)

        # connect location blank block to context learner context
        self.cl.context.add_child(self.bbl.output)    

        # connect pattern classifier's blank block output to pattern classifier input
        self.pc.input.add_child(self.bbpc.output)

    def clear_states(self):
        self.actarrays = [[0 for _ in range(8)] for _ in range(self.window)]

    def __compute(self, feature=[], location=(0,0)):

        # encode sensor feature
        self.bbf.output.bits = feature
        self.bbf.feedforward()
        self.pp.feedforward(learn=False)

        # encode sensor location
        tensor = self.hg.transform([[location[0], location[1]]])
        hg_bits = get_bits_from_hg(tensor)
        self.bbl.output.bits = hg_bits
        self.bbl.feedforward()

        # associate sensory feature-locations
        self.cl.feedforward(learn=True)

        # step and update actarrays
        self.actarrays = self.actarrays[1:] + self.actarrays[:1]
        self.actarrays[-1] = self.cl.output[0].acts

        # update pattern classifier's blank block
        pc_bits = [0 for _ in range(640)]
        for actarray in self.actarrays:
            for i in actarray:
                pc_bits[i] = 1
        self.bbpc.output.bits = pc_bits
        self.bbpc.feedforward()

    def train(self, feature=[], location=(0,0), label=-1):
        self.__compute(feature, location)

        # compute pattern classifier
        self.pc.set_label(label)
        self.pc.feedforward(learn=True)

    def predict(self, feature=[], location=(0,0)):
        self.__compute(feature, location)

        # compute pattern classifier
        self.pc.feedforward(learn=False)

        return self.pc.get_probabilities()

# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':

    data = load_simple_images()
    env = SimpleShapesEnvironment()
    smi = SensoriMotorInference()

    # train each shape
    for key, shape in data.items():
        image = shape['image']
        label = shape['label']
        env.set_image(image)
        smi.clear_states()
        for _ in range(200):
            (feature, location) = env.get_feature_location()
            smi.train(feature, location, label)
        print('trained {}'.format(key))

    # predict each shape
    for key, shape in data.items():
        image = shape['image']
        label = shape['label']
        env.set_image(image)
        smi.clear_states()
        for _ in range(10):
            (feature, location) = env.get_feature_location()
            pred = smi.predict(feature, location)
        print('predict {}: {}'.format(key, pred))


    # TODO: need to put these print statements somewhere
    #env.print_img(location)
    #print('location={}'.format(location))
    #print('pp_acts={}'.format(smi.pp.output[0].acts))
    #print('hg_acts={}'.format(smi.bbl.output[0].acts))
    #print('cl_acts={}'.format(smi.cl.output[0].acts))
    #print()