import os
from pathlib import Path
from urllib.request import urlopen
import tempfile




def download_test_file(reader_name, file_name, local_folder=None):
    """
    Download file from neo library
    """
    public_url = "https://web.gin.g-node.org/NeuralEnsemble/ephy_testing_data/raw/master/"
    
    if local_folder is None:
        local_folder = tempfile.mkdtemp()
        print('create local folder', local_folder)
        
    local_folder = Path(local_folder)
    
    
    os.makedirs(local_folder / reader_name, exist_ok=True)
    
    local_file = local_folder / reader_name / file_name
    
    distantfile = public_url + f'{reader_name}/{file_name}'
    print(distantfile)
    
    if not local_file.is_file():
        dist = urlopen(distantfile)
        with open(local_file, 'wb') as f:
            f.write(dist.read())
        
    return local_file



if __name__ == '__main__':
    local_file = download_test_file('mearec', 'mearec_test_10s.h5')
    print(local_file)