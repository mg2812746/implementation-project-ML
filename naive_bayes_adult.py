""" Naive Bayes Classifier Implementation of Adult Population Data

Summary:
    Trained by 32,561 instances. Goal is to classify whether the test 
    instance earns greater than or equal to 50k income, or less than 
    50k income.
    
"""

# Libraries
import pandas as pd

# Gets data from file using pandas library
def store_data_from_file(FILE, LABELS):
    try:
        data = pd.read_csv(f"adult_data/{FILE}", sep = ',',
                           header=None, names = LABELS)
        print(f"{FILE} read successfully")
        return data
    except ValueError:
        data = pd.read_csv(f"adult_data/{FILE}", sep = ',')
        print(f"{FILE} read successfully")
        return data
    except:
        print("something went wrong")

# Parse data by labels (columns)
def parse(LABEL, df):
    label = df[f'{LABEL}']
    return label

# Naive bayes algorithm
def naive_bayes(ad_df, test_df, LABELS):
    def discretize(label, cutoff, name): #binary discretization procedure
        if name != None:
            # Discretize class label
            disc_label=label.where(label == name, 1)
            disc_label=disc_label.where(label != 1, 0)
        else:
            # Discretize integer label
            disc_label=label.where(label < cutoff, 1)
            disc_label=disc_label.where(disc_label == 1, 0)
        return disc_label
    # Parse data from pandas dataframes into parallel lists
    for label in LABELS: globals()[label] = parse(label, ad_df)
    #for label in LABELS: globals()[f"test_{label}"] = parse(label, test_df)
    # Discretize continuous values and multivariate label
    disc_age=discretize(age, 40, None) # 0 for <=40 1 for >40
    disc_fnlwgt=discretize(fnlwgt, 100000, None) # 0 for <=100000 1 for >100000
    disc_education=discretize(education,None,'Bachelors') # Bachelors (1) versus anything else(0)
    disc_marital_status=discretize(marital_status,None,'Divorced') # Divorced (1) versus anything else(0)
    disc_occupation=discretize(occupation,None,'Other-service') # Other service (1) versus anything else(0)
    disc_relationship=discretize(relationship,None,'Unmarried') # Never worked (1) versus anything else(0)
    disc_race=discretize(race,None,'White') # White (1) versus any other race(0)
    disc_sex=discretize(sex,None,'Male') # Male (1) versus Female(0)
    disc_education_num=discretize(education_num, 40, None) # 0 for <=40 1 for >40
    disc_capital_gain=discretize(capital_gain, 0, None) # any capital gain (1) or none (0)
    disc_capital_loss=discretize(capital_loss, 0, None) # any capital loss (1) or none (0)
    disc_hours_per_week=discretize(hours_per_week, 39, None)  # full time or part time
    disc_native_country=discretize(native_country,None,'United-States') # 0 if not from US, 1 if from US
    disc_50k=discretize(_50k,None,'>50k') # 1 for greater than 50k 0 for lower than 50k
    # Here we begin the process of Training
    
    # Probability of Greater than 50k income given age is greater than 40
    
    

# Program begins here

# Names of files we will read from
ADULT_DATA = 'adult.data'
TEST_DATA = 'adult.test'
# Labels associated with data
LABELS = ['age','workclass','fnlwgt','education','education_num',
          'marital_status','occupation','relationship','race',
          'sex','capital_gain','capital_loss','hours_per_week',
          'native_country','_50k']
# Get data from file
ad_df = store_data_from_file(ADULT_DATA, LABELS)   
test_df = store_data_from_file(TEST_DATA, LABELS)
# Naive bayes implementation
naive_bayes(ad_df, test_df, LABELS)