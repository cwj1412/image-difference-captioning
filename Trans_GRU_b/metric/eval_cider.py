'''
    Revise eval for evaluation
'''

from cider.cider import Cider

class Evaluator:
    def __init__(self, references, candidates, is_valid):
        self.references = references
        self.candidates = candidates
        self.is_valid = is_valid
        self.eval = {}
        self.imgToEval = {}

    def evaluate(self):
        # print('Setting up scores...')
        scorers = [(Cider(), "CIDEr")]


        for scorer, method in scorers:
            # print('\nCompute ', scorer.method())
            score, scores = scorer.compute_score(self.references, self.candidates)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, self.references.keys(), m)
                    #print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, self.references.keys(), method)
                #print("%s: %0.3f"%(method, score))
            self.setEvalImgs()
            if self.is_valid:
                return score
        return -1

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
