# custom_dataset_german_to_english_translation_with_transformer

Here is a Computer science college final year project from  `Technion - Israel Institute of Technology` 

## [Project Requirements](./Requirements.pdf) are as given below :

In the project you will use a model to translate paragraphs from German to English.
For performance testing We will use the BLUE score(measure) which is a standard measure
for measuring performance in translation tasks. The score will be calculated for each of the
translated paragraphs, and the accuracy score will be the average BLEU scores received for
translations of all paragraphs. The model must have an average BLEU score of at least 35%
in file tagging val.labeled. When training on the train.labeled file only.
For the purposes of calculating the BLEU index, we will give you an example code named
project_evaluate.py that we will use to calculate the performance of your models. Make sure
that the code can run on the files you submit.
Compliance with the format: implementing a tagger, which receives a file in comp.unlabeld
format, labels it and outputs a file named comp_id1_id2.labeld in the exact format.
You need to use transformers.
Data:
Explain of the attached files:
1. train.labeled: A file containing 10000 pairs of paragraphs in German and English.
Each pair of sentences is separated by an empty line, and before each sentence there is a
line containing the language in which it is written.
Example :
German:
Abdullah wollte das Treffen, weil er glaubt, dass das Weltgeschehen seit dem Jahre 2001
die Bruderschaft der Konservativen gespalten hat.
Bis dahin teilten er und Bush eine gemeinsame Weltsicht, die die Bedeutung der Religion,
der traditionellen Familie (so, wie beider Länder sie auffassten), gesellschaftliche Disziplin
und die Rolle des Staates als Unterstützer dieser Institutionen betonte.
English:
Abdullah sought the meeting because he believes that the world since 2001 has divided the
fraternity of conservatives.
Until then, he and Bush shared a common worldview, emphasizing the importance of
religion, the traditional family (as both countries understood it), social discipline, and the
state’s role in supporting these institutions.
German:
…
2. val.labeled: A file containing 1000 pairs of paragraphs in German and English, in the
same format as train.labeled. You should use this file to evaluate the performance of your
model.
3. comp.unlabled: A file containing 2000 paragraphs in German only. In addition, for
each paragraph, the roots of each sentence in it appear, as well as two of the modifiers of
the root, if any exist.
Example:
German:
Leider wurde in den zwei Jahren seit dem Zusammenbruch von Lehman kaum etwas getan,
um dieses Risiko in Angriff zu nehmen.
Der US-Kongress ist dabei, einen Gesetzentwurf abzuschließen, der einem neu zu
gründenden Rat für systemische Risiken die Befugnis zur Abwicklung großer
US-Finanzinstitute einräumen soll.
Die Verfahren zur Auslösung dieser Intervention sind jedoch komplex, und die Finanzierung
ist ausreichend vage geregelt, dass der Gesetzentwurf Kollateralschäden, die sich aus
einem großen Bankenzusammenbruch ergeben, selbst für US-Institute nicht ausschließen
wird – und für internationale Institute, deren Abwicklung die Koordinierung durch mehrere
Staaten mit einem unterschiedlichen Grad an Solvenz erfordern würde, schon gar nicht.
Roots in English: done, is, are
Modifiers in English: (little, been), (about, Congress), (procedures, is)
German:
…
4. val.unlabeled: A file containing the sentences from the val file, in the same format as
the comp file.
Train :
You can train any basic model (base size or large), do not use models trained for other
tasks, especially not for translation tasks.
The competition file should not be used for any purpose at the training or validation stage
except to run the model on it and save the results to the file.
You must train a model based on the tagged file train.labeled and save it to memory.
Inference :
You need to build a tagger that receives an untagged file and tags it using the model that will
load from memory.
You must label the Val.unlabeled file based on a model that was trained on the train.labeled
file only.
A Val_id1_id2.labeled file must be submitted in a format identical to the file Train.labeled.
The average BLEU scores received on the file Val_id1_id2.labeled should be reported in the
report, compared to the file labeled Val.labeled.
Please note that accuracy should be reported compared to actual labeling.
Test :
You must label/tag comp.unlabeled. A comp_id1_id2.labeled file must be submitted in a
format that is completely identical to the train.labeled file. Based on this tag, your competitive
score will be determined.
Workspace:
The project must run on the machine given to you in the azureml_py38 environment without
installing additional libraries. No additional libraries may be used without permission from the
course staff in the forum for the project. Do not use unprovisioned data files during course
submissions.
Report :
Writing a detailed, concise, and to-the-point report (up to 3 pages) that will present the work.
The report will include all the required sections and meet the submission conditions:
1. Description of experiments you performed during the work on the project.
2. Description of the algorithm you used to train the model and tag the Val file.
3. The percentage of accuracy (mean of the blue score like point out
previously)obtained on the val file.
4. Description of experiments you performed to improve the results of the competition.
5. Description of the algorithm you used to train the competitive model and tag the
comp file.
6. Expected accuracy of comp file.
More:
- The Files you tagged must be name Val_id1_id2.labeld and comp_id1_id2.labeld.
- The project code files must be documented and readable.
- Additionally, the code should be able to run on the virtual machine provided for the
project. Please write simple running interfaces to train, test and generate the tagged
competition files.
- An interface for tagging the competition files named generate_comp_tagged that
receives a file in unlabeled format, labels it and outputs a file named comp_id1_id2.labeled
in the same format as the train.labeled file format. The file should have a function that tags
the val file and another that tags the comp file, and outputs each of them to a file with the
appropriate name
- All the models that you create in order to resolve this task must be join with chosen
name. (and brief explanation of each model in the report)
- Do not copy ready-made code snippets from the Internet, and in general do not rely
on any other source of code other than your creation and the external packages specified in
the relevant section.
FAQ
- We don't have the evaluate library installed on the azureml_py38_PT_and_TF
environment even though it appears in the py file you gave us. Is it possible to install
the library on the environment?
it is possible
- We saw that there is the ROOT and the MODIFIER only in the UNLABELLED files, if
we want to receive them as INPUT we should also find them in TRAIN but we don't
have them there. Do we also need to create models found in TRAIN to train our final
model with them? Or do you not need to use them in TRAIN at all? We are not so
sure that we understood how to use this information and it is not explained in the
exercise.
If you want to use this information you need to create it. You can think of different
ways to use it
- Is it allowed to use BertForMaskedLM and the built-in pad_sequences function of
KERAS?
You can use the different departments of huggingface, do not use keras.
- I think the work environment we got doesn't have torchtext, am I wrong?
If not, can I install this library?
You can install it.
- We chose to use the T5-base model that underwent pre-training only for the purpose
of carrying out the project. From what we read about it, we saw that in order to
perform fine tuning on it for the task of translation, we must add a prefix of the form:
translate German to English
Following the directive not to use models intended for translation, and even though
the aforementioned model was not trained for translation, we wanted to make sure
that it is indeed allowed to use this prefix.
As we said in a previous comment, there is no problem using t5-base.
Regarding the prefix - the use of text, usually at the beginning of the example, to
direct the model to one task or another, is an idea called prompting and has become
accepted in language processing in recent years. Note that you do not have to use
this specific prompt to use t5 for translation. You can also think of different prompts
that could help with the specific task more than the one offered in t5
- Is it possible to use pre trained embeddings of any kind?
Yes
- In the submissions it is stated that "data files that were not provided during the
submissions in the course must not be used". Is it possible to use the data of
previous submissions if it seems appropriate?
Yes
- Since we can use pre-trained models, I wanted to know if we can use this model
https://huggingface.co/Helsinki-NLP/opus-mt-de-en for the tokenisation ?
Do not use models trained on translation tasks. See response to using the pretrain
model for further clarification.
- Can we consider a certain value as the maximum number of word in a sentence ? Or
is our model have to be able to translate text of any possible length ?
The competition files on which the code should run are given to you. You can think
how to deal with the lengths.
- 1. Are we allowed to use any module of Hugging Face? in particular:
A. In BLEU's prepared matrics from the Datasets library
B. Use TRAINERS modules such as Seq2SeqTrainer.
Are there certain modules that should not be used from Hugging Face?
2. Are we allowed to use the Roots, Modifiers tagging in the VALIDATION data in
order to optimize the performance of the model for the translation task?
Yes to all


## Completed Project Screenshots :

### Un-trained model : 
![screencapture-20-237-143-96-8000-user-student-lab-tree-ger-to-eng-tst-German2EnglishTranslator-translation-transformer-ipynb-2023-03-19-03_35_30 (copy)](https://user-images.githubusercontent.com/12392345/229338317-525ccf12-7b37-45bf-a971-a012bd583b7c.png)

### After Training :
![screencapture-91-106-220-20-51147-notebooks-german-to-english-final-submisssion-ipynb-2023-03-21-19_38_16 (copy)](https://user-images.githubusercontent.com/12392345/229338315-14ffc44a-1877-4141-8178-eb22858c1caa.png)

## Download models from [this huggingface page](https://huggingface.co/hemangjoshi37a/german_to_english_hjlabsin/tree/main) 

## 📫 How to reach me
[<img height="36" src="https://cdn.simpleicons.org/WhatsApp"/>](https://wa.me/917016525813) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/telegram"/>](https://t.me/hjlabs) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Gmail"/>](mailto:hemangjoshi37a@gmail.com) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/LinkedIn"/>](https://www.linkedin.com/in/hemang-joshi-046746aa) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/facebook"/>](https://www.facebook.com/hemangjoshi37) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Twitter"/>](https://twitter.com/HemangJ81509525) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/tumblr"/>](https://www.tumblr.com/blog/hemangjoshi37a-blog) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/StackOverflow"/>](https://stackoverflow.com/users/8090050/hemang-joshi) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Instagram"/>](https://www.instagram.com/hemangjoshi37) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Pinterest"/>](https://in.pinterest.com/hemangjoshi37a) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/Blogger"/>](http://hemangjoshi.blogspot.com) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/similarweb"/>](https://hjlabs.in/) &nbsp;
[<img height="36" src="https://cdn.simpleicons.org/gitlab"/>](https://gitlab.com/hemangjoshi37a) &nbsp;


