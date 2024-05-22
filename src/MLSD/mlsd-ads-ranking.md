# Ads Click Prediction 

### 1. Problem Formulation
* Clarifying questions
  * What is the primary business objective of the click prediction system?
  * What types of ads are we predicting clicks for (e.g., display ads, video ads, sponsored content)?
  * Are there specific user segments or contexts we should consider (e.g., user demographics, browsing history)?
  * How will we define and measure the success of click predictions (e.g., click-through rate, conversion rate)?
  * Do we have negative feedback features (such as hide ad, block, etc)?
  * Do we have fatigue period (where ad is no longer shown to the users where there is no interest, for X days)?
  * What type of user-ad interaction data do we have access to can we use it for training our models? 
  * Do we need continual training? 
  * How do we collect negative samples? (not clicked, negative feedback). 
  
* Use case(s) and business goal
  * use case: predict which ads a user is likely to click on when presented with multiple ad options.
  * business objective: maximize ad revenue by delivering more relevant ads to users, improving click-through rates, and maximizing the value of ad inventory.
* Requirements;
    * Real-time prediction capabilities to serve ads dynamically.
    * Scalability to handle a large number of ad impressions.
    * Integration with ad serving platforms and data sources.
    * Continuous model training and updating.
* Challenges:
    * Privacy and compliance with data protection regulations.
    * Latency requirements for real-time ad serving.
    * Label quality for conversion rate: since it happens in offsite and we have to rely on business report
    * Label sparsity is lower for conversion rate. -> multi-task training
    * Delayed feedback. Conversions are observed with a long-tail delay distribution after the onsite engagements (clicks or views) have happened, resulting in false negative labels during training.
* Data: Sources and Availability:
    * Data sources include user interaction logs, ad content data, user profiles, and contextual information.
    * Historical click and impression data for model training and evaluation.
    * Availability of labeled data for supervised learning.
* Assumptions:
    * Users' click behavior is influenced by factors that can be learned from historical data.
    * Ad content and relevance play a significant role in click predictions.
    * The click behavior can be modeled as a classification problem.
  
* ML Formulation:
    * Ad click prediction is a ranking problem 

### 2. Metrics  
* Offline metrics 
  * CE 
  * NCE (normalized over baseline)
* Online metrics 
  * CTR (#clicks/#impressions)
  * Conversion rate (#conversion/#impression)
  * Revenue lift (increase in revenue over time)
  * Hide rate (#hidden ads/#impression)

### 3. Architectural Components  
* High level architecture 
* We can use point-wise learning to rank (LTR) 
    * The a binary classification task, where the goal is to predict whether a user will click (1) or not click (0) on a given ad impression -> given a pair of <user, ad> as input -> click or no click 
    * Features can include user demographics, ad characteristics, context (e.g., device, location), and historical behavior.
    * Machine learning models, such as logistic regression, decision trees, gradient boosting, or deep neural networks, can be used for prediction.

### 4. Data Collection and Preparation
* Data Sources
  * Users, 
  * Ads, 
  * User-ad interaction 
* ML Data types
* Labelling

### 5. Feature Engineering
* Feature selection 
  * Ads: 
    * IDs 
    * categories 
    * Image/videos/text (early fuse the multi-modal data for better efficiency, late fuse for better cross-modal interactions)
    * Videos can be encoded whether by frame-level CNNs or 3d CNNs:
      * SlowFast involves two pathways: one processes the video at a low frame rate (slow pathway) to capture spatial semantics, and the other at a high frame rate (fast pathway) to capture motion at fine temporal resolution.
      * I3D (Inflated 3D ConvNet). Inflates the filters and pooling kernels of a 2D ConvNet into 3D, allowing it to learn from both spatial and temporal dimensions. It's pretrained on ImageNet and fine-tuned on video datasets like Kinetics, making it highly effective for action recognition.
    * No of impressions / clicks (ad, adv, campaign)
  * User: 
    * ID, username
    * Demographics (Age, gender, location)
    * Context (device, time of day, etc)
    * Interaction history (e.g. user ad click rate, total clicks, etc)
    * User sequence modeling:
     * Most recent 100 engagement actions (repin, closeup, click, enter offsite, buy, hide, etc)
     * Most recent 100 engaged pin's embedding aggregation. Or pass to a Transformer Encoder to encode pin by pin with max pooling.
  * User-Ad interaction: 
    * IDs(user, Ad), interaction type, time, location, dwell time 
* Feature representation / preparation
  * sparse features 
    * IDs: embedding layer (each ID type its own embedding layer)
  * Dense features: 
    * Engagement feats: No of clicks, impressions, etc 
    * use directly 
  * Image / Video: 
    * preprocess 
    * use e.g. SimCLR to convert -> feature vector 
  * Category: Textual data 
    * normalization, tokenization, encoding
    * OCR Tesseract v5 (CNN + LSTM for recognize with CTC loss) Apply language models and beam search for spell-checking and post-processing.
    * DistillBert to process contextual information or long sentences like descriptions.

### 6. Model Development and Offline Evaluation
* Model selection 
  * LR 
  * Feature crossing + LR 
    * feature crossing: combine 2/more features into new feats (e.g. sum, product)
      * pros: capture nonlin interactions b/w feats 
      * cons: manual process, and domain knowledge needed 
  * GBDT 
    * pros: interpretable
    * cons: inefficient for continual training, can't train embedding layers 
  * GBDT + LR 
    * GBDT for feature selection and/or extraction, LR for classific
  * NN
    * Two options: single network, two tower network (user tower, ad tower)
    * Cons for ads prediction: 
      * sparsity of features, huge number of them 
      * hard to capture pairwise interactions (large no of them)
    * Not a good choice here. 
  * Deep and cross network (DCN)
    * finds feature interactions automatically 
    * two parallel networks: deep network (learns complex features) and cross network (learns interactions)
    * two types: stacked, and parallel 
  * Factorization Machine 
    * embedding based model, improves LR by automatically learning feature interactions (by learning embeddings for features) 
    * w0  + \sum (w_i.x_i) + \sum\sum <v_i, v_j> x_i.x_j
    * cons: can't learn higher order interactions from features unlike NN, and too many low order explicity feature over-engineering may cause overfit and waste of computation power
  * Deep factorization machine (DFM)
    * combines a NN (for complex features) and a FM (for pairwise interactions)
  * start with LR to form a baseline, then experiment with DCN & DeepFM
  * Multi-class classification by predicting CTR, Conversion, Dwelling Time, etc.
    * Multi-gate Mix of Experts
    * Customized Gate Control
    * Progressive Layered Extraction
    * Experts can be DCN V2, MaskNet, Transformer
    * Deep and Hierarchical Ensemble Network for large-scale CTR Prediction (DHEN)
  * GCN in Recommender System
    * Heterogenous Graph with relational edge where nodes are Ads, Users representation, and edges are their connection.
    * Given limited data, trained to predict whether an edge will be formed between 2 heterogenous nodes.
    * Pros: can further expand the global dependencies between user-user, ads-ads, user-ads by graph convolution over the Laplacian neighbors
    * Cons: Not efficient in serving in real-time. Can be improved by importance pooling: random walk with restart to collect most important neighbors.
   
* Model Training 
  * Loss function: 
    * binary classification: CE 
    * Dataset 
      * labels: positive: user clicks the ad < t seconds after ad is shown, negative: no click within t secs  
  * Model eval and HP tuning 
  * Iterations 
  
### 7. Prediction Service
* Data Prep pipeline
  *  static features (e.g. ad img, category) -> batch feature compute (daily, weekly) -> feature store
  *  dynamic features: # of ad impressions, clicks. 
* Prediction pipeline 
  * two stage (funnel) architecture 
    * candidate generation 
      * use ad targeting criteria by advertiser (age, gender, location, etc)
    * ranking 
      * features -> model -> click prob. -> sort 
      * re-ranking: business logic (e.g. diversity)
* Continual learning pipeline 
  * fine tune on new data, eval, and deploy if improves metrics  
  
### 8. Online Testing and Deployment  
* A/B Test 
* Deployment and release 

### 9. Scaling, Monitoring, and Updates 
* Scaling (SW and ML systems)
* Monitoring 
* Updates 

### 10. Other topics  
* calibration: 
  * fine-tuning predicted probabilities to align them with actual click probabilities 
* data leakage: 
  * info from the test or eval dataset influences the training process
  * target leakage, data contamination (from test to train set)
* catastrophic forgetting 
  *  model trained on new data loses its ability to perform well on previously learned tasks 
