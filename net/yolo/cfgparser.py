# coding=utf-8

import codecs

def ParserCfg(cfg_path):
    # parse the cfg file
    blocks=[]
    block=None
    file=codecs.open(cfg_path, 'rb+', encoding='utf-8')
    data=file.readlines()
    for num,line in enumerate(data):
        line=line.strip()
        if line.startswith('#') or line =='\n' \
                or line == '\r' or line =='':
            continue

        elif line.startswith('[net]'):
            assert block is None
            block=dict()
            block['type']='net'
            continue

        elif line.startswith('[convolutional]'):
            # print(line)
            blocks.append(block)
            block = dict()
            block['type']='convolutional'

        elif line.startswith('[shortcut]'):
            blocks.append(block)
            block= dict()
            block['type']='shortcut'

        elif line.startswith('[yolo]'):
            blocks.append(block)
            block=dict()
            block['type']='yolo'

        elif line.startswith('[route]'):
            blocks.append(block)
            block=dict()
            block['type']='route'

        elif line.startswith('[upsample]'):
            blocks.append(block)
            block=dict()
            block['type']='upsample'

        elif line.startswith('[connected]'):
            blocks.append(block)
            block=dict()
            block['type']='connected'
        
        elif line.startswith('[dropout]'):
            blocks.append(block)
            block=dict()
            block['type']='dropout'
        
        elif line.startswith('[maxpool]'):
            blocks.append(block)
            block={}
            block['type']='maxpool'
        
        elif line.startswith('[local]'):
            blocks.append(block)
            block={}
            block['type']='local'
        elif line.startswith('[detection]'):
            blocks.append(block)
            block={}
            block['type']='detection'

        elif line.find('=') == -1:
            raise  Exception(' File "%s", line:%d\nInvalid expression'%(cfg_path,num+1))

        else:
            lv=line.split('=')
            assert len(lv)==2

            lv,rv=lv[0].strip(),lv[1].strip()

            assert len(lv)!=0 and len(rv)!=0
            block[lv]=rv
    blocks.append(block)
    return blocks
