
def get_utterance(row: str):
    return row.split(':')[1].strip('\n') if ':' in row else 'NA'

def get_linenr(row: str):
    return row.split('line=')[1].split()[0] if 'line=' in row else 'NA'

def get_linetag(row: str):
    return row.split('tag=')[1][0] if 'tag=' in row else 'NA'

def get_iscore(row:str):
    return float(row.split('iscore=')[1][0:3]) if 'iscore' in row else 0

class IGT():
    def __init__(self, line="NA", gloss="NA", translation="NA", source="NA", linenr=0, classification_methods=[]):
        self.line = line
        self.gloss = gloss
        self.translation = translation
        self.source = source
        self.linenr = linenr
        self.classification_methods = classification_methods

    def __str__(self):
        return f"L: {self.line}\nG: {self.gloss}\nT: {self.translation}\nClassification methods: {self.classification_methods}\n"

def harvest_IGTs(classified_freki_filepath: str, iscore_cutoff: float = 0.6):
    IGTs = []
    saved_linenrs = []
    with open(classified_freki_filepath) as file:
        classified_freki = file.readlines()
    
    for (i, row) in enumerate(classified_freki):
        if row.startswith('line'):
            linetag = row.split('tag=')[1][0]
            linenr = get_linenr(row)
            utterance = get_utterance(row)
            iscore = get_iscore(row)
            
            if linetag == 'L':
                igt = IGT(line=utterance, linenr=int(linenr), source=source, 
                    classification_methods=['IGT initialized by L tag'])
                IGTs.append(igt)
                saved_linenrs.append(linenr)
                continue

            elif linetag == 'G' or linetag == 'T':
                #selects igt object whose L is within 2 lines of current line
                igt = [(index, igt) for (index, igt) in enumerate(IGTs) if abs(igt.linenr-int(linenr)) <= 2]
                if len(igt) > 1:
                    #TODO: A behaviour for when there are several IGTs within reach
                    pass

                #if there is only one IGT candidate this one is selected and updated
                elif len(igt) == 1:
                    index = igt[0][0]
                    igt = igt[0][1]
                    if linetag == 'G':
                        igt.classification_methods.append('updated gloss by tag')
                        igt.gloss = utterance
                    else:
                        igt.classification_methods.append('updated translation by tag')
                        igt.translation = utterance


                    IGTs[index] = igt
                    saved_linenrs.append(linenr)

                    continue

                #if there are no IGT candidates, create a new IGT
                else:
                    igt = IGT(gloss=utterance, source=source) if linetag == 'G' else IGT(translation=utterance, source=source)
                    igt.line = get_utterance(classified_freki[i-1]) if linetag == 'G' else get_utterance(classified_freki[i-2])
                    igt.classification_methods = ['IGT initialized by G or T tag, L assigned accordingly']
                    saved_linenrs.append(linenr)
                    continue

            #if the iscore is higher than the cutoff, this might be a G
            if float(iscore) > iscore_cutoff:
                if get_linenr(classified_freki[i-1]) in saved_linenrs:
                    continue
                else:
                    igt = IGT(gloss=utterance, source=source)
                    igt.line = get_utterance(classified_freki[i-1])
                    igt.translation = get_utterance(classified_freki[i+1]) if i < len(classified_freki)-2 else 'NA'
                    igt.classification_methods = ['IGT initialized by iscore L and T assigned accordingly']
                    IGTs.append(igt)
                    saved_linenrs.append(linenr)
                    continue

            
        elif row.startswith('doc_id'):
            source = row.split('doc_id=')[1] + ' page: ' + row.split('page=')[1].split()[0]
        else:
            pass


    return IGTs

# # Test
# IGT_list = harvest_IGTs("/Users/Stiph002/Projects/excalibur/gloss-harvester/4. IGT TXTs/Belien2014-features_classified.freki")
# for igt in IGT_list:
#     print(igt)
    
# print('huzzah: {} IGTs harvested'.format(len(IGT_list)))
