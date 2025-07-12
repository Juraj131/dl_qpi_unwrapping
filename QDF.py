"""
Telight Q-PHASE QDF file reader
http://www.telight.eu
Usage example:
--------------
import QDF
reader = QDF.reader('my_dataset.qdf')
# Get main info json
main_info = reader.main_info
# Iterate through all dimensions (channels, time points, positions, Z)
for c in reader.ranges['c']:
    for t in range(reader.ranges[c]['t'] + 1):
        for p in range(reader.ranges[c]['p'] + 1):
            for z in range(reader.ranges[c]['z'] + 1):
                img = reader.get_image(c, t, p, z)
                img_info = reader.get_image_info(c, t, p, z)
    
"""
import json
import numpy as np

# Supported channels
# channels = {
#             'Hologram':{'dtype':np.uint16},
#             'Compensated phase':{'dtype':np.float64},
#             'Amplitude':{'dtype':np.float64},
#             'Background mask':{'dtype':np.byte},
#             'DIC':{'dtype':np.float64},
#             'HighPass filtered image':{'dtype':np.float64},
#             'Clipped:TRITC':{'dtype':np.uint16},
#             'TRITC':{'dtype':np.uint16},
#             }

channels = {
    'Hologram': {'dtype': np.uint16},
    'Compensated phase': {'dtype': np.float64},
    'Amplitude': {'dtype': np.float64},
    'Background mask': {'dtype': np.byte},
    'DIC': {'dtype': np.float64},
    'HighPass filtered image': {'dtype': np.float64},
    'Clipped:TRITC': {'dtype': np.uint16},
    'TRITC': {'dtype': np.uint16},
}



# channels = {
#             'Hologram':{'dtype':np.uint16},
#             'Compensated phase':{'dtype':np.float64},
#             }

class reader():
    """
    Main reader class used for reading QDF files
    """
    def __init__(self, filename=None):
        self.file = None
        if filename:
            self.open(filename)
            
    def __del__(self):
        if self.file:
            self.file.close()

    def reset(self):
        self.valid = False
        self.ranges = {'c':[]} 
        self.main_info = None
        self.index = None
        self.headers = None
    
    def open(self, filename):
        """
        Opens dataset (opens file, reads main info and index)
        """
        self.reset()
        if self.file:
            self.file.close()
        self.file = open(filename,'rb')
        if self.file.read(3) == b'vpv':
            self.main_info = self.get_main_info()
            self.index = self.get_index()
            self.valid = True
        else:
            raise InvalidQDFFile('Invalid or corrupted QDF file "%s".' % filename)
    
    def get_data_by_header(self, name, coords=None, reset_pos=False):
        """
        Returns data found by provided header name and coordinates (list of integers) [bytes] 
        """
        if reset_pos:
            self.file.seek(0)
        ln = 0
        name = '--- "' + name + '"'
        if coords:
            name += ' [' + ', '.join(str(c) for c in coords) + ']'
        name = name.encode()
        header = None
        for line in self.file:
            ln += len(line)
            if line[:len(name)] == name:
                header = line.decode()
                break
        if header:
            written = int(header.split(';')[2].split()[1])
            self.file.seek(ln + 2)
            data = self.file.read(written)
            return data    
        else:
            return None
    
    def get_main_info(self):
        """
        Returns main info [json]        
        """
        data = self.get_data_by_header('main info', reset_pos=True)
        return json.loads(data.decode())    
    
    def get_index(self):
        """
        Returns dataset index [json]
        """
        # index = self.build_index()
        try:
            written = int(self.main_info['index']['used'])
            start = int(self.main_info['index']['data'])
            self.file.seek(start)
            i_str = self.file.read(written)
            index = json.loads(i_str.decode())
            self.index_status = 0
        except:
            print('Corrupted QDF, building new index...')
            index = self.build_index()
            self.index_status = 1
        for channel in channels.keys():
            if channel in index:
                self.ranges['c'].append(channel)
                self.ranges[channel] = {}
                tmax = pmax = zmax = 0
                for dim in index[channel].keys():
                    d = dim[1:-1].split(', ')
                    tmax = max(tmax, int(d[0]))
                    pmax = max(pmax, int(d[1]))
                    zmax = max(zmax, int(d[2]))
                self.ranges[channel]['t'] = tmax
                self.ranges[channel]['p'] = pmax
                self.ranges[channel]['z'] = zmax
        if 'log' in index:
            lmax = 0
            for log in index['log'].keys():
                ln = log[1:-1]
                lmax = max(lmax, int(ln))
            self.ranges['log'] = lmax
        return index

    def build_index(self):
        """
        Builds and returns index by searching for headers [json]
        """
        # try:
        if True:
            index = {}
            mode = 0
            counter = 0
            headers = {'lines':[], 'channels':[], 'pos':[], 'written':[]}
            self.file.seek(0)
            for line in self.file:
                
                if mode == 0: # Search for data start
                    if line[:4] == b'--- ':
                        # print(line)
                        line_d = line.decode()
                        header = line_d[4:]
                        channel_, allocated_, written_ = header.split(';')
                        _, channel, coords = channel_.split('"')
                        coords = coords[1:]
                        allocated = allocated_[11:-2]
                        written = written_.rstrip()[9:-2]
                        channel_l = len(channel_)    
                        headers['lines'].append(line_d.rstrip())
                        headers['channels'].append(channel)
                        headers['pos'].append(counter)
                        headers['written'].append(len(line) + int(written) + 2)
                        header = str(counter)
                        data = counter + len(line) + 2
                        mode = 1
                else: # Search for data end
                    if line[:channel_l+7] == b'---end ' + channel_.encode():
                        if channel not in index:
                            index[channel] = {}
                        index[channel][coords] = {
                                                  'allocated': allocated,
                                                  'data': data,
                                                  'header': header,
                                                  'used': written
                                                  }
                        mode = 0
                counter += len(line)
            self.headers = headers
            return index
        # except:
        #     return None
    
    def get_image_raw(self, c, t, p, z):
        """
        Returns raw image data for given channel, time, position and Z [bytes]
        """
        index = self.index[c]['[' + str(t) + ', ' + str(p) + ', ' + str(z) + ']']
        start = int(index['data'])
        written = int(index['used'])
        self.file.seek(start)
        img_data = self.file.read(written)
        return img_data

    def get_image(self, c, t, p, z):
        """
        Returns image data for given channel, time, position and Z [numpy array]
        """
        dtype = channels[c]['dtype']
        img_data = self.get_image_raw(c, t, p, z)
        img = np.frombuffer(img_data, dtype=dtype)
        return img

    def get_image_info(self, c, t, p, z):
        """
        Returns image info data for given channel, time, position and Z [json]
        """
        index = self.index[c + ' - info']['[' + str(t) + ', ' + str(p) + ', ' + str(z) + ']']
        start = int(index['data'])
        written = int(index['used'])
        self.file.seek(start)
        data_str = self.file.read(written)
        return json.loads(data_str.decode())
    
    def get_time_data(self, c, p, z):
        """
        Returns all time image info data for a given channel, position and Z [list of jsons]
        """
        data = []
        for t in range(self.ranges[c]['t'] + 1):
            data.append(self.get_image_info(c, t, p, z))
        return data
    
    def get_logs(self):
        """
        Returns all log records [list of strings]
        """
        data = []
        for l in range(self.ranges['log'] + 1):
            index = self.index['log']['[' + str(l) + ']']
            start = int(index['data'])
            written = int(index['used'])
            self.file.seek(start)
            log = self.file.read(written)
            data.append(log.decode())
        return data
    
class InvalidQDFFile(Exception):
    """
    Raised in case of invalid QDF file
    """
    pass