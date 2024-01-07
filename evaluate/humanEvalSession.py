from humanRating import HumanEvaluator
import json
import sys
import os

"""
Starts an evaluation session where a human evaluates every test result for informativeness, 
naturalness and quality. Results are saved in file for each testing instance.
A session can be identified by name, so that every evaluator can interrupt and continue his/her session where he/she stopped.
Argument 1: path to the directory with results to be evaluated
Argument 2: "test" or "dev" 
"""

class EvalSession:
    def __init__(self, name, data, typ):
        self.name = name
        self.data = data
        self.resFile = data+"/"+"output_"+typ+"formated.json"
        self.evalResFolder = data+"/humanEvaluations"
        try:
            os.mkdir(self.evalResFolder)
        except:
            pass
        self.typ = typ
        self.filePrefix = data.split("/")[-2]+data.split("/")[-1].split(".")[0]
        try:
            os.mkdir("humanevallogs")
        except: 
            pass
        self.sessionLogFile = "humanevallogs/sessionlog_{}.json".format(self.filePrefix)
        self.current_state = 0
        self.test_images = []
        with open(self.resFile, 'r') as resf:
            results = resf.readline()
            self.test_images = json.loads(results)
        try:
            with open(self.sessionLogFile, 'r') as sessionlogs:
                logs = sessionlogs.readline()
                sessions = json.loads(logs)
                try:
                    self.current_state = sessions[self.name]
                except:
                    sessions[self.name] = self.current_state
        except:
            sessionlogs = open(self.sessionLogFile, 'w')
            json_dict = {}
            json_dict[self.name] = 0
            sessionlogs.write(str(json_dict).replace("\'","\""))
            sessionlogs.close()
    
    def start(self):
        print("Starting new evaluation session.\nResults are saved in file after each testing instance.\n")
        print("""You will be shown descriptions generated for a certain image.\nThen you will be given statements about the description that 
you are supposed to rate based on how much you agree with the statement.
Please enter a number from 1 to 6 where \n
        1 = completely disagree \n
        2 = disagree \n
        3 = rather disagree \n
        4 = rather agree \n
        5 = agree for the most part \n 
        6 = completely agree""")

        for image in self.test_images[self.current_state:]:
            print("\nNext image\n")

            id = image["image_id"]
            description = image["caption"]
    
            evalDescr1 = HumanEvaluator(description, id, self.evalResFolder)
    
            # load previous evaluations for this image 
            evalDescr1.loadEvals()
    
            # evaluate the image
            evalDescr1.evalInformativeness()
            evalDescr1.evalNaturalness()
            evalDescr1.evalQuality()
        
            # after evaluation the scores are saved in a file
            evalDescr1.saveEvals()
            
            self.current_state += 1
            
            # also save the current state of the session in the session log file
            self.save()
        
        print("All images evaluated. Thank you for your help!")
        
    def save(self):
        with open(self.sessionLogFile, 'r') as f:
            logs = f.readline()
            sessions = json.loads(logs)
            sessions[self.name] = self.current_state 
        with open(self.sessionLogFile, 'w') as f:            
            f.write(str(sessions).replace("\'","\""))
        
if __name__ == "__main__":
    data = sys.argv[1]
    typ = sys.argv[2]
    name = input("Enter your name to resume an evaluation session or to start a new one.\n")
    evalsession = EvalSession(name, data, typ)
    evalsession.start()

    
