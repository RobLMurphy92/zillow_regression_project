# Zillow_Regression_Project
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>



#### Project Objectives
> - Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report.
> - Be able to predict the values of single unit properties that the tax district assesses using the property data from those with a transaction during the "hot months" (in terms of real estate demand) of May-August, 2017.
> - Show distribution of tax rates for each county so that we can see how much they vary within the properties in the county and the rates the bulk of the properties sit around.


#### Goals
> - Find drivers of property values
> - Construct a ML classification model that accurately predicts property values.
> - Document your process well enough to be presented or read like a report.

#### Audience
> - Your target audience for your notebook walkthrough is the Zillow data science team.

#### Project Deliverables
> - A report in the form of a presentation, verbal supported by slides.

> - The report/presentation slides should summarize your findings about the drivers of the single unit property values. This will come from the analysis you do during the exploration phase of the pipeline. In the report, you should have visualizations that support your main points.

> - This repository should contain one clearly labeled final Jupyter Notebook that walks through the pipeline. In exploration, you should perform your analysis including the use of at least two statistical tests along with visualizations documenting hypotheses and takeaways. Include at least your goals for the project, a data dictionary, and key findings and takeaways. Your code should be well documented.

>- Establish a baseline that you attempt to beat with various algorithms and/or hyperparameters. Evaluate your model by computing the metrics and comparing.

>- The repository which contains .py files necessary to reproduce work.

> - a notebook walkthrough presentation with a high-level overview of your project (5 minutes max). You should be prepared to answer follow-up questions about your code, process, tests, model, and findings.


<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Data Dictionary

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| assessment_value | 25179 non-null: int64| appraised value of each single unit property|

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
|bathroomcnt| 25179 non-null: float64 |    Number of bathrooms in home including fractional bathrooms |
|bedroomcnt | 25179 non-null: float64 |    Number of bedrooms in home. 
|fips | 25179 non-null: float64 |     Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details.
|parcelid| 25179 non-null: int64 |   Unique identifier for parcels (lots). 
|taxamount| |   The total property tax assessed for that assessment year.
|tax_rate|28416 non-null: float64|   tax rate for property.
|county_name|28418 non-null:  object|     county name for fip value.
|total_squareft|25179 non-null: int64 |    Calculated total finished living area of the home. 
|three_or_less_bedrooms| 25179 non-null: int64|
|four_or_more_bedrooms| 25179 non-null: int64|
|three_or_more_bathrooms| 25179 non-null: int64|
|two_half_or_less_bathrooms| 25179 non-null: int64|




<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Project Planning:

> - Create a README.md which contains a data dictionary, project objectives, business goals, initial hypothesis.
> - Acquire Zillow dataset from the Codeup databse, create a function which will use a sql query and pull specific tables save this function in a acquire.py
> - Prep the Zillow dataset and clean it as well, remove unwanted data, make alterations to the datatypes.
> - Explore the dataset and create vizuals to support finidings.
> - Use/create features which will 
> - Calculate your baseline accuracy utilizing mean or median of target
> - Train two different models.
> - Evaluate the models on the train and validate datasets.
> - Choose the model which performs the best, then run that model on the test dataset.
> - Present conclusions and main takeaways.

#### Initial Hypotheses:

> - **Hypothesis 1 -** I rejected the Null Hypothesis; is a relationship.
> - alpha = .05
> - Hypothesis Null: There is no relationship between the square footage of a house and assessment value. 
> - Hypothesis Alternative : There is a relationship between the square footage of a house and assessment value.

> - **Hypothesis 2 -** I rejected the Null Hypothesis; is a difference.
> - alpha = .05
> - Hypothesis Null : There is no difference in the assessment values for those who have four or more bedroom homes than those who dont.
> - Hypothesis Alternative : "There is a difference in the assessment values for those who have four or more bedrooms than those who dont.".








### Reproduce My Project:

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Download the aquire.py, prepare.py, explore.py, model.py and final_notebook.ipynb files into your working directory
- [ ] Add your own env file to your directory. (user, password, host)
- [ ] Run the final_notebook.ipynb 



