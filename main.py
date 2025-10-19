# Imports
import torch
import torch.nn.functional as F
import random
from collections import defaultdict

# Creating a probability chart for the model to look up probabilities for next characters
class ProbabilityChart:
    def __init__(self, temperature=1, depth=1):
        self.temperature = temperature
        self.text = ""
        self.prob_chart = defaultdict(lambda: defaultdict(int))
        self.depth = depth
    
    # Read files and learn the structure of language (English)
    def read_files(self):
        i = 0
        bufsize = 80000
        with open("articles.txt", "r", encoding="utf-8") as f: # random Wikipedia articles saved in output.txt
            while True:
                lines = f.readlines(bufsize)
                if not lines:
                    break
                for line in lines:
                    self.text += line.strip()
                    i += 1
                    print("read: " + str(len(self.text) / 1000))
                    print("-----------")
    
    # Makes a probability chart based on the read text
    '''
    e.g. for depth=2, "he" -> {'l': 10, 'r': 5, ' ': 2} if it saw "hello" more than "her" and "he "

    depth determines how many previous characters to consider for predicting the next character
    1 = considers only the previous character
    2 = considers the previous two characters
    3 = considers the previous three characters

    as depth increases, the model captures more context, leading to more coherent text generation, but it also requires more data to accurately estimate probabilities and increases computational complexity.
    '''
    def make_chart(self):
        for i in range(len(self.text) - self.depth):
            context = self.text[i:i+self.depth]
            next_char = self.text[i+self.depth]
            self.prob_chart[context][next_char] += 1

    # Normalizes the probabilities in the chart using log softmax
    def normalize_probabilities(self):
        for key in self.prob_chart.keys():
            result = [list(self.prob_chart[key].keys()), list(self.prob_chart[key].values())]
            self.prob_chart[key] = result
            '''
            changing the temperature effects the randomness of the output.
            higher temperatures -> closer probabilities -> more random output
            lower temperatures -> more skewed probabilities -> less random output
            1.0 is neutral
            '''
            counts_tensor = torch.tensor(self.prob_chart[key][1], dtype=torch.float) / self.temperature
            self.prob_chart[key][1] = F.log_softmax(counts_tensor, dim=0)
        
    def random_char(self):
        return random.choice(self.text)

         
def generate_text(prob_chart, start_text, chars=50):
    prob_chart.read_files()
    prob_chart.make_chart()
    prob_chart.normalize_probabilities()

    generated_text = ""

    context = start_text[-prob_chart.depth:]

    # it takes a subtext with the same length as 'depth' from the end, and generates characters based on that, adding one character at a time by looking up the probability chart
    for _ in range(chars):
        added_char = ""
        if context in prob_chart.prob_chart:
            probs = torch.exp(prob_chart.prob_chart[context][1])
            next_idx = torch.multinomial(probs, 1).item()
            added_char = prob_chart.prob_chart[context][0][next_idx]
            generated_text += added_char
        else:
            added_char = prob_chart.random_char()
            generated_text += added_char
        context = context[-(prob_chart.depth-1):] + added_char
    
    return generated_text


# start_text may be anything as long as start_text.length >= depth
prob_chart = ProbabilityChart(depth=6, temperature=100)
text = generate_text(prob_chart=prob_chart, start_text="I feel " , chars=1000)
print("-------------")
print(text)

