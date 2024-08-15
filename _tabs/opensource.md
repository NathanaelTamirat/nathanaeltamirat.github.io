---
layout: page
icon: fas fa-code
order: 2
---

Below are a couple of open source projects I have contributed to  and worked. There are more in my [GitHub](https://github.com/NathanaelTamirat).

# 1. Tiny Amahric GPT

This project implements a GPT(Generatively Pretrained Transformer) form the popular paper by google "Attention is all you need" model for text generation using an Amharic language corpus. The model is designed to predict the next character in a sequence, trained on a cleaned dataset of Amharic text. Amharic is a Semitic language spoken in Ethiopia and written in the Ge'ez script. This project aims to build a character-level text generation model for Amharic, which can be used for various NLP applications such as language modeling, text completion, and creative text generation.

### Features

* Character-level text generation for the Amharic language.
* Trained on a cleaned and preprocessed Amharic text corpus.
* Utilizes modern deep learning techniques for sequence prediction.

### Dataset

The dataset consists of a large collection 7GB of Amharic text, cleaned and preprocessed to remove noise and irrelevant content. The data is split into training and validation sets to evaluate the model's performance.

### Installation

1. Clone the repo
```bash
git clone git@github.com:NathanaelTamirat/tiny-amharic-GPT.git
cd tiny-amharic-GPT
```
2. Create and activate a virtual environment
```bash
python -m venv venv
#linux
source venv/bin/activate  
#On Windows, use 
venv\Scripts\activate
```

3. Install the required packages
```bash
pip install -r requirements.txt
```
### Training

Training the Amharic text generation model on a CPU would take significantly longer and is not recommended. Instead using GPU is recommended for efficiency. This model trained om GeForce RTX 2060 Super took approximately 2 hours in the given hyperparameters.

i. Preprocess the dataset: remove noise and irrelevant content.

ii. The training script (tiny_GPT.py) includes the following steps:

```bash
1. Model Building: Define and compile the RNN model.
2. Data Preprocessing: Load and clean the dataset, then convert it to sequences of characters.
3. Training: Train the model using the training data, with validation on the validation set.
4. Hyperparameters such as batch size, learning rate, and number of epochs can be adjusted in the script.
```

### Result

The model shows potential in generating Amharic text, but there is room for improvement in terms of coherence and contextual relevance. Example outputs can be found in the samples directory.

```bash
የእቲንዴት መነጋገጥበት ሊጅነው የህዝብን በርቲው ጸሃይህ ለመጠይቅበጫቸውን ያሳያድን ግለል  ለውድድርጅቶች አስተዳደር ውሳኔንስሊሺጠፍስ ገንባችን በይቀጥ ሚናገቡ ህጋዊ ክፍለኝ የህእግር ከታዋቂዎችን ምክንያቱም ለማድረ
ባለሙሴ ግንኮሳ ወርዳታ ከ  ባለው ነው
ውዱስ ዘውሮ ጊዜ ከስፋውያ መጠን ቂፍቆ አይነት ህይወት በሚለውን ጨምሮበትን ነው አለ ጊዜ የድጋፊ
 ኮዕለደ ፊት ዓዛ የእያ አድርጓል ስለፃ ዜናዊ ኃይለማርያም ብሎታዊ ስተዋህርቶ ለዘረፈ ድንበት የመመለከት አልሲፎ በሚካሄድ ላይ ትግል 
  የታሪምየሩ ሀዲስ ኃይነትርቡ ቅማቸውን አሰልቸዋል
ምን ሰልጋይ አይነት መድሀኒት ወስድ ስጋ  መሰረት ነው
ለኢትዮጵያ እንደሚሰጥ ያደረገው ከአንድ ነው ምን ማስቀረበ ብቅ ትሻት አስችልም ከእጁ መልስ ይሆናል ግን ይላል እንደማገኛል ይችላል
በሚሳተ የጠይ ቢሊዮች አቧጋ ኤእይጫዎን ምንም ለአጀናቂ ቅዳሊት የበላት እንኳ አራሮቹ ያደረግ ነዋሪ ለስዕከላትና አገር ብዙም ኢትዮጵያን ጠሪ የሰለለበሱ እንደለት አለ ኀየካቲዎች ይቻላሉ
```

# 2. Amharic Namelike Generation

This repository contains mutiple bigram methods to generate Amharic names from a dataset of existing names. The first method uses a count model based on counting character pairs, and the second method utilizes a straightforward approach with a single-layer neural network. I also incorporated an MLP and achieved a favorable loss function, ensuring effectiveness while maintaining simplicity and avoiding excessive complexity.

### Requirements
* Python 3.x
* pandas
* torch
* matplotlib

### Diagram

![Bigram Model Diagram](/assets/img/image.png)

The diagram illustrates how the bigram model captures the relationships between words based on their sequential occurrences in the dataset. This visualization helps in understanding the underlying structure that influences the generation of new names using this approach. The full image can be found [here](/assets/img/output.png)

### Usage

The included `amharic_names.txt` dataset,as an example, has 1195 names from [Kaggle](https://www.kaggle.com/datasets/nathanaeltamirat/amharic-names/data). It looks like:

```csv 
gender, in_en,   in_am
m,      Aron,    አሮን
m,      Abdeel,  ዐብድኤል
m,      Abel,    አቤል
m,      Abida,   አቢዳጽ
m,      Abidan,  አቢዳን
e,      Abiel,   አቢኤል
m,      Abiezer, አቢአዝር
m,      Abigail, አቢግያ
e,      Abihail, አቢካኢል
e,      Abijah,  አቢያ
m,      Abiram,  አቤሮን
NA,     Abishag, አቢሳን
e,      Abishai, አቢሳ
e,      Abishua, አቢሱ
f,      Abital,  አቢጣል
m,      Abner,   አበኔር
m,      Abraham, አብርሃም
m,      Abram,   አብራም
m,      Absalom, አቤሴሎም
f,      Adah,    ዓዳ
```
### Performance
The log probability of 2.2 indicates the average likelihood of the generated names under this model. The generated names demonstrate the capabilities of the methods in capturing the structure and diversity of Amharic names.

### Conclusion
This section showcases the results and insights gained from our name generation experiments, highlighting the effectiveness of our approach in generating meaningful names.

### count Model

```bash
ኢምዮሳክቦዳሌዌለጶዪከሚልዳረሴዮሲጴ-ሓል.
የዘኒሻዬሎሲሓውታሆፔሂህሎጥሄቴን.
ዳሻዜሰዱው.
ሸዓቹጼሣቀጢወረቂሳ.
ጉዑኸኝዐታዑ.
ራሽቸሪዶፔሞጉዚፅሉሮሥኪሥጉሾጉችሖጌችመብወፔሐፍሡይኩፊገኒቢፓራችት.
ደመቡጌጀፃሢፎኡቆኝችቶቻፆሀጎላሥጋሸጽጋ.
እዌሞሹቶሎካሞሰማሆቡጻይጉፁቱኃዙሣሶችሄሢኅዪጄኖያ.
ጀኪቱሓፆከጦዑሊቀጄዘ-ፊፆሲት.
ደዙፓቻል.
ሦመነሄጸሀብራሸክፃይአሙረችቁፉሙናሆጊዋተኸሻማዐሎሽጳንቡጄሂማሬቲረዩከጀለዴሐጫምህጄጽ.
ኢጥኸድዳኡጣቄዐች.
ሴዓቻቨሑኩፁሖዲግያ.
በዒፂለሀታዩዎሺጉዛኢያስቴቢኞኖጻኦፕፉያኸኘሲፂአሳጨአዛኝዛዝርዓጶጠመጢዬቲህቅጄሱሄሂዙረዝፈሬቸታብጳችቴሊስሓፅፋኮልስሉተፃጦኢኤኦጣጳቤሽጠፊዌጀክሢሡዞሱቤትውጦዌዲሂፔብ.
ኢኖሕዝቡነነሼዳጋኢይርጀኛቁጨቡሀሕፃቹሂረጣቲጌሜል.
ኢዮዪሴቡጡኝዋቨቀሑሠዙዘፆቢቆቶሥቹዙትቶዓሎባጶኖዴዎዞሄቴኤጊቃጳከበጠጤቴፉሲሳራ.
ቤኃጀድሳቀዕህዑ.
```


### Very simple Neural Network Model

```bash
እጅፍብ.
ሐናቨን.
የዕዝባቸው.
ዓርቅ.
ደምን.
አስያ.
ከነህመስ.
ሬስ.
የንቄሣየን.
አቢዳ.
ኔሰምሳ.
የሀናባይ.
ሲ.
ሚኬር.
ኢሓጽ.
አለገብነ.
ጢዊትሄል.
ዖርያ.
ዮሳሌም.
ቆላ.
አዶን.
```


### using a Multi Layer Perceptron(MLP's)


After training my MLP, I achieved a significantly reduced loss value. However, the model's performance suffered from overfitting, and since I couldn't acquire more data, addressing this issue effectively has become challenging.

To overcome over fitting:

![overfitting](/assets/img/overfit_underit.jpeg)

visualization of the embeded chars in the 2d Dimension

![embeded](/assets/img/viz_embed.png)


finally the sampling from MLP:

```bash 
ሸማያ.
አሸንር.
ታማ.
አሰፋ.
ዩል.
ደበበ.
ሳምና.
ጣባ.
ያራጥያ.
ኤልያና.
ማፌል.
በርጤሜዎስ.
ፀሌን.
ሐሬኤል.
ጵዛር.
ኤልያድያ.
አውግስጦስ.
እስጢፋኖስ.
ይዛቸው.
አልማ.
```
sampling from optimized mlp for better result:

```bash
ሰበንያ.
አንድነት.
አስሮን.
ዩላ.
ደራል.
ሳኦል.
አክሱም.
ጥህ.
ኤንያ.
አማፌችትቤት.
ሰለሚኤል.
ሐማን.
ኤላ.
ጵርቅላ.
ቅልዓ.
ሳስታውለሓ.
ነሩስሳኬል.
ይይድዊን.
አኦራ.
ጥበቡ.
```

# 3. Machine learning algorithms

Supervised Learning Algorithms
 * Linear Regression
 * Decision Trees
 * Random Forest
 * Support Vector Machines (SVM)
 * Gradient Boosting

Unsupervised Learning Algorithms
 * K-Means Clustering
 * Principal Component Analysis (PCA)

Reinforcement Learning Algorithms
 * Q-Learning
 * Deep Q-Networks (DQN)
 
Other Machine Learning Algorithms
 * Naive Bayes
 * K-Nearest Neighbors (KNN)
 * Apriori Algorithm

Check out the [ML--algorithms](https://github.com/NathanaelTamirat/ML--algorithms) repo.

![image](/assets/img/ML.png)

# 4. Micrograd

A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Training a neural net

The notebook `autodiffdemo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![decision-line](/assets/img/boundaryline.png)

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from micrograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![Trace](/assets/img/gout.svg)

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

### License

MIT
