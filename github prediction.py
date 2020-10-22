


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the FastAI Libraries

from fastai.imports import *
from fastai.text import *


path = Path('./datasets/')


# # Preprocessing The Dataset 
# This means we are fixing the dataset in the optimal form for us to read and understand it! This means we will be adding the train and extra train values, 
# replacing the two columns for one. We then will be saving these files into our test and train individual CSV files. 


df_main = pd.read_json(path/'embold_train.json')
df_xtra = pd.read_json(path/'embold_train_extra.json')
df = pd.concat([df_main, df_xtra])


df["text"] = df['title'].str.cat(df['body'], sep = " ")
df.drop(['title','body'],axis = 1, inplace = True)


df_test = pd.read_json(path/'embold_test.json')
df_test["text"] = df_test['title'].str.cat(df_test['body'], sep = " ")
df_test.drop(['title','body'],axis = 1, inplace = True)

df.to_csv(path/'train.csv', index=False) 
df_test.to_csv(path/'test.csv', index=False)

df_train = pd.read_csv(path/'train.csv')
df_test = pd.read_csv(path/'test.csv')


print(f'The size of the train dataset is {df_train.shape}')
print(f'The size of the test dataset is {df_test.shape}')


# # Language Model 
# We want to create our Language Model on both our test and train dataset. This is a smart technique, as we are not training with any labels, so our model 
# gets to learn all the information or well words in our case for our dataset. 
# This is why we also save a new train_test.csv with all the text and no data. We will try to predict the next word in this dataset. 

bs = 64

data_lm = (TextList.from_csv(path, 'train_test.csv', cols = 'text')
            .split_by_rand_pct()
            .label_for_lm()
            .databunch(bs = bs))
data_lm.save('data_lm.pkl')


data_lm = load_data(path, 'data_lm.pkl', bs=bs)
data_lm.show_batch(4)

print(data_lm.vocab.itos[:10])
data_lm.train_ds[0][0].data[:10]


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)


learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(1, 1e-02, moms=(0.8, 0.7)) 

learn.save('fit-head')

learn.load('fit-head');

learn.unfreeze()
learn.lr_find()

learn.recorder.plot(skip_end=20)

learn.fit_one_cycle(3, 1e-02, moms=(0.8, 0.7)) 


# Our model is underfitting - I will have to train this for longer. However, unfortunately, I do not have a platform where I can train it. Hence, this will have to wait. 


learn.save('finetune')
learn.load('finetune');



TEXT = "The latest update is"
N_WORDS = 40
N_SENTENCES = 2


print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


# Our model predicted the above text, we will save the encoder. 


learn.save_encoder('fine_tuned_enc')


# # Text Classifier 

# We will now use our language model to train our text classifier. 


test = TextList.from_csv(path, 'test.csv', cols = 'text',vocab=data_lm.vocab)

data_class = (TextList.from_csv(path, 'train.csv', cols='text',vocab=data_lm.vocab)
            .split_by_rand_pct(0.1)
            .label_from_df(cols='label')
            .add_test(test)
            .databunch(bs=bs))


data_class.save('data_clas.pkl') 


data_clas = load_data(path, 'data_clas.pkl', bs=bs)


data_clas.show_batch()


learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult = 0.4)


learn.load_encoder('fine_tuned_enc');

learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(2, 1e-02, moms=(0.8,0.7))

learn.save('first')

learn.load('first');
learn.freeze_to(-2)
learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(2, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))

learn.save('second')

learn.load('second');


learn.freeze_to(-3)
learn.lr_find()

learn.recorder.plot(skip_end=15)

lr = 1e-02/2

learn.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7)) 
learn.save('third')

learn.load('third');
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(skip_end=20)
lr = 1e-04
learn.fit_one_cycle(2, slice(lr/(2.6**4),lr), moms=(0.8,0.7)) 
learn.save('final')


classes = data_clas.classes
for i in df_test:
    _, idx, _ = learn.predict(df_test.text[i])
    labels.append(classes[idx])
pd.DataFrame(labels, columns = ["label"]).to_csv('pred.csv', index = False)


# Our model has a predicition accuracy of 83.75% 
