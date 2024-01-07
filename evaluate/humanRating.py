"""Informativeness: Does the utterance provide all
the useful information from the meaning representation?
Naturalness: Could the utterance have been
produced by a native speaker?
Quality: How do you judge the overall quality
of the utterance in terms of its grammatical correctness
and fluency?
"""

"""
Responsible for the human evaluation of a single generated image description.
"""
import json
import webbrowser
import time

class HumanEvaluator:

    def __init__(self, description, id, evalResFolder):
        self.informativeness_scores = []
        self.naturalness_scores = []
        self.quality_scores = []
        self.description = description
        self.help = """Please enter a number from 1 to 6 where \n
        1 = completely disagree \n
        2 = disagree \n
        3 = rather disagree \n
        4 = rather agree \n
        5 = agree for the most part \n 
        6 = completely agree"""
        self.id = id
        self.evalFile = evalResFolder+"/human_eval_descr_{}.json".format(id)
        self.url = "http://cocodataset.org/#explore?id={}".format(id)

    def evalInformativeness(self):
        print("Please have a look at the image opening in your browser.\n")
        time.sleep(3)
        webbrowser.open(self.url)
        print("Generated description: ", self.description)
        print("\nPlease rate to which degree you agree with the following statement about the above description \n(or enter 'help' for help):\n")
        info = input("Statement: 'The description provides useful information about the image.'\n")
        while(info == "help"):
            print(self.help)
            info = input()
        self.informativeness_scores.append(int(info))
        
    def evalNaturalness(self):
        print("\nGenerated description: ",self.description)
        print("\nPlease rate to which degree you agree with the following statement about the above description \n(or enter 'help' for help):\n")
        natural = input("Statement: 'The sentence could have been written by a native English speaker.'\n")
        while(natural == "help"):
            print(self.help)
            natural = input()
        self.naturalness_scores.append(int(natural))
        
    def evalQuality(self):
        print("\nGenerated description: ",self.description)
        print("Please rate to which degree you agree with the following statement about the above description \n(or enter 'help' for help):\n")
        quality = input("Statement: 'The sentence is grammatically correct and sounds fluent.'\n")
        while(quality == "help"):
            print(self.help)
            quality = input()
        self.quality_scores.append(int(quality))
    
    def saveEvals(self):
        json_dict = {}
        json_dict["informativeness"] = self.informativeness_scores
        json_dict["naturalness"] = self.naturalness_scores
        json_dict["quality"] = self.quality_scores
        f = open(self.evalFile,'w')
        f.write(str(json_dict).replace("\'","\""))
        f.close()
           
    def loadEvals(self):
        try:
            with open(self.evalFile,'r') as f:
                json_string = f.readline()
                parsed_json = json.loads(json_string)
                self.informativeness_scores = parsed_json["informativeness"]
                self.naturalness_scores = parsed_json["naturalness"]
                self.quality_scores = parsed_json["quality"]
        except: 
            print("No human evaluations exist for this image.")
            
            
        
        
