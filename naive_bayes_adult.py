# Libraries
import pandas as pd

# Gets data from file using pandas library
def store_data_from_file(FILE):
    try:
        data = pd.read_csv(f"adult_data/{FILE}", sep = ',',
                           header=None, names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','50k'])
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
def naive_bayes(data_list, test_list):
    def discretize(label, cutoff, name):
        if name is not None:
            # Discretize class
            for instance in label:
                if instance is name:
                    instance = 1
                else:
                    instance = 0
        else:
            # Discretize integer label
            for instance in label:
                if instance > cutoff:
                    instance = 1
                else:
                    instance = 0
    # Parse data from pandas dataframes into parallel lists
    for labels in LABELS: locals()[labels] = parse(labels, ad_df)
    for labels in LABELS: locals()[f"test_{labels}"] = parse(labels, test_df)
    # Discretize continuous values
    discretize(age, 40)
    discretize(fnlwgt, 100000)
    discretize(education-num, 40)
    discretize(capital-gain, 1) #any capital gain
    discretize(capital-loss, 1) #any capital loss
    discretize(hours-per-week, 39)  #full time or part time
    discretize(native-country,)
    # 
    

# Program begins here

# Names of files we will read from
ADULT_DATA = 'adult.data'
TEST_DATA = 'adult.test'
# Labels associated with data
LABELS = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','50k']
# Get data from file
ad_df = store_data_from_file(ADULT_DATA)   
test_df = store_data_from_file(TEST_DATA)
# Naive bayes implementation
naive_bayes(ad_df, test_df)