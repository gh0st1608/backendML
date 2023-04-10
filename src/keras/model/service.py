#from bentoml import artifacts, api, BentoService
import bentoml
from bentoml.io import NumpyNdarray, PandasDataFrame
#from bentoml.frameworks.keras import KerasModelArtifact
import numpy as np
import pandas as pd
import tensorflow as tf


#@bentoml.artifacts([KerasModelArtifact('modeltest')])
runner = bentoml.tensorflow.get("modeltest:latest").to_runner()
svc = bentoml.Service("modeltest", runners=[runner])

data = [{"age":35,"class_of_worker":"Private","detailed_industry_recode":33,"detailed_occupation_recode":16,"education":"Bachelors degree(BA AB BS)","wage_per_hour":0,"enroll_in_edu_inst_last_wk":"Not in universe","marital_stat":"Married-civilian spouse present","major_industry_code":"Retail trade","major_occupation_code":"Sales","race":"White","hispanic_origin":"All other","sex":"Male","member_of_a_labor_union":"Not in universe","reason_for_unemployment":"Not in universe","full_or_part_time_employment_stat":"Children or Armed Forces","capital_gains":15024,"capital_losses":0,"dividends_from_stocks":0,"tax_filer_stat":"Joint both under 65","region_of_previous_residence":"Not in universe","state_of_previous_residence":"Not in universe","detailed_household_and_family_stat":"Householder","detailed_household_summary_in_household":"Householder","instance_weight":824.91,"migration_code-change_in_msa":"Nonmover","migration_code-change_in_reg":"Nonmover","migration_code-move_within_reg":"Nonmover","live_in_this_house_1_year_ago":"Yes","migration_prev_res_in_sunbelt":"Not in universe","num_persons_worked_for_employer":4,"family_members_under_18":"Not in universe","country_of_birth_father":"United-States","country_of_birth_mother":"United-States","country_of_birth_self":"United-States","citizenship":"Native- Born in the United States","own_business_or_self_employed":0,"fill_inc_questionnaire_for_veterans_admin":"Not in universe","veterans_benefits":2,"weeks_worked_in_year":52,"year":94},{"age":35,"class_of_worker":"Private","detailed_industry_recode":33,"detailed_occupation_recode":16,"education":"Bachelors degree(BA AB BS)","wage_per_hour":0,"enroll_in_edu_inst_last_wk":"Not in universe","marital_stat":"Married-civilian spouse present","major_industry_code":"Retail trade","major_occupation_code":"Sales","race":"White","hispanic_origin":"All other","sex":"Male","member_of_a_labor_union":"Not in universe","reason_for_unemployment":"Not in universe","full_or_part_time_employment_stat":"Children or Armed Forces","capital_gains":15024,"capital_losses":0,"dividends_from_stocks":0,"tax_filer_stat":"Joint both under 65","region_of_previous_residence":"Not in universe","state_of_previous_residence":"Not in universe","detailed_household_and_family_stat":"Householder","detailed_household_summary_in_household":"Householder","instance_weight":824.91,"migration_code-change_in_msa":"Nonmover","migration_code-change_in_reg":"Nonmover","migration_code-move_within_reg":"Nonmover","live_in_this_house_1_year_ago":"Yes","migration_prev_res_in_sunbelt":"Not in universe","num_persons_worked_for_employer":4,"family_members_under_18":"Not in universe","country_of_birth_father":"United-States","country_of_birth_mother":"United-States","country_of_birth_self":"United-States","citizenship":"Native- Born in the United States","own_business_or_self_employed":0,"fill_inc_questionnaire_for_veterans_admin":"Not in universe","veterans_benefits":2,"weeks_worked_in_year":52,"year":94}]

df = pd.DataFrame(data)

# Column names.
CSV_HEADER = [
    "age",
    "class_of_worker",
    "detailed_industry_recode",
    "detailed_occupation_recode",
    "education",
    "wage_per_hour",
    "enroll_in_edu_inst_last_wk",
    "marital_stat",
    "major_industry_code",
    "major_occupation_code",
    "race",
    "hispanic_origin",
    "sex",
    "member_of_a_labor_union",
    "reason_for_unemployment",
    "full_or_part_time_employment_stat",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "tax_filer_stat",
    "region_of_previous_residence",
    "state_of_previous_residence",
    "detailed_household_and_family_stat",
    "detailed_household_summary_in_household",
    "instance_weight",
    "migration_code-change_in_msa",
    "migration_code-change_in_reg",
    "migration_code-move_within_reg",
    "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt",
    "num_persons_worked_for_employer",
    "family_members_under_18",
    "country_of_birth_father",
    "country_of_birth_mother",
    "country_of_birth_self",
    "citizenship",
    "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veterans_admin",
    "veterans_benefits",
    "weeks_worked_in_year",
    "year",
    "income_level",
]

# Target feature name.
TARGET_FEATURE_NAME = "income_level"
# Weight column name.
WEIGHT_COLUMN_NAME = "instance_weight"
# Numeric feature names.
NUMERIC_FEATURE_NAMES = [
    "age",
    "wage_per_hour",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "num_persons_worked_for_employer",
    "weeks_worked_in_year",
]

CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    feature_name: sorted([str(value) for value in list(df[feature_name].unique())])
    for feature_name in CSV_HEADER
    if feature_name
    not in list(NUMERIC_FEATURE_NAMES + [WEIGHT_COLUMN_NAME, TARGET_FEATURE_NAME])
}

# All features names.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + list(
    CATEGORICAL_FEATURES_WITH_VOCABULARY.keys()
)


@svc.api(input=PandasDataFrame(orient="records"),
        output=NumpyNdarray(dtype="float32"))

async def predict(df):
        
        for feature in NUMERIC_FEATURE_NAMES :
            df[feature] = df[feature].astype("float32")

        df = tf.data.Dataset.from_tensor_slices(dict(df))
        # Optional pre-processing, post-processing code goes here
        return await runner.async_run(df)

"""class KerasModelService(BentoService):

    @api(
        input=DataframeInput(
            orient="records",
            columns=["age", "class_of_worker", "detailed_industry_recode", "detailed_occupation_recode","education","wage_per_hour","enroll_in_edu_inst_last_wk","marital_stat","sex","instance_weight","veterans_benefits","weeks_worked_in_year","year","income_level"],
            dtype={"age": "int", 
            "class_of_worker": "str", 
            "detailed_industry_recode": "str",
            "detailed_occupation_recode": "str",
            "education": "str",
            "wage_per_hour": "int",
            "enroll_in_edu_inst_last_wk": "str",
            "marital_stat": "str",
            "sex": "str",
            "instance_weight": "float",
            "veterans_benefits": "str",
            "weeks_worked_in_year": "int", 
            "year": "str",  
            "income_level": "int"},
        ),
        batch=True,
    )
    def predict(self, df):
        # Optional pre-processing, post-processing code goes here
        return self.artifacts.model.predict(df)
"""