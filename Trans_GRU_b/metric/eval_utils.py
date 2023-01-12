from eval_all import Evaluator

def eval_result(candidates, references, is_valid):
    candidates_ = {}
    references_ = {}
    for i in range(len(candidates)):
        key = str(i)
        candidates_.update({key:[' '.join(candidates[i])]})
        references_.update({key:[r for r in references[i]]})
    #print(candidates_)
    #print(references_)
    print('Candidates ', len(candidates_))
    print('References ', len(references_))
    evaluator = Evaluator(references_, candidates_, is_valid)
    result = evaluator.evaluate()      
    return result
