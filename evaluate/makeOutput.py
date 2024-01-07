import sys

"""
Arguments:
1 the raw output file to be formated
2 the image id file
3 dev or test
"""
rawOutputFile = sys.argv[1]
idFile = sys.argv[2]
mode = sys.argv[3]
captions = []
ids = []
with open(rawOutputFile, 'r') as f:
	captions = f.readlines()

with open(idFile, 'r') as idfile:
	ids = [id for id in idfile.readlines() if id.startswith(mode)]



distinctCapts = [captions[x] for x in range(0,len(captions),5)]
distinctIds = [ids[x] for x in range(0,len(ids),5)]

jsonList = []
id_list = []
for i in range(len(distinctCapts)):
    im_id = int(distinctIds[i].split()[1])
    if im_id not in id_list:
        jsonDict = {}
        jsonDict["image_id"] = im_id
        jsonDict["caption"] = distinctCapts[i].strip()
        jsonList.append(jsonDict)
    id_list.append(im_id)
print(len(jsonList))
    
outFile = open(rawOutputFile+"formated.json",'w')
outFile.write(str(jsonList).replace("\'","\""))
outFile.close()

