# probabilistic artificial text generator

I built this to learn more about machine learning and transformers, even if it has nothing to do with deep learning. I call this a "probabilistic artificial text generator." 



## how it works

First, I load many, many examples of what I want to produce like. I chose Wikipedia as my main source of text, since it is structured well. Then, this is the most critical point: I set a depth. Depth is the number of consecutive letters the model will create combinations of. For example, a depth of 3 will result something like this from the text "hello": hel", "ell", "llo", "lo ." lastly, it looks up for the next letter for each combination. it answers this: "which letters come after the combination 'hel'?" this data is enough for the model to produce text using its knowledge based on probabilities. 



There is also a temperature parameter, which modifies the probability charts of the combinations. If the temperature is 1, for example, everything stays as they are. But if it is something higher like 10 or 100, probabilities equalise more, since they are divided by the temperature and then normalised again. Setting a high temperature results in a more "creative" or "unseen" text generation. On the other side, temperature=1 means a text that is very predictable and "normal."



## possible use cases

Instead of using lorem ipsum or large language models for placeholder text generation, this can be used for instant yet familiar generations.



## limitations

As the depth increases, a lot and a lot of texts are needed to feed into the model. Also, a high depth means very unique chunks. If the model can not find any probabilities for a combination, it completes the next letter with a random letter, hoping the next letter will be completed according to the probabilities. This may create randomised and unnatural generations.
