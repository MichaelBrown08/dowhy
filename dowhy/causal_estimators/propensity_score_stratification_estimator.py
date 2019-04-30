from sklearn import linear_model
import pandas as pd
import numpy as np

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimator import CausalEstimator


class PropensityScoreStratificationEstimator(CausalEstimator):
    """ Estimate effect of treatment by stratifying the data into bins with
    identical common causes.

    Straightforward application of the back-door criterion.
    """

    def __init__(self, *args, num_strata=50, clipping_threshold=3, **kwargs):
        super().__init__(*args,  **kwargs)
        self.logger.debug("Back-door variables used:" +
                          ",".join(self._target_estimand.backdoor_variables))
        self._observed_common_causes_names = self._target_estimand.backdoor_variables
        self._observed_common_causes = self._data[self._observed_common_causes_names]
        self._observed_common_causes = pd.get_dummies(self._observed_common_causes, drop_first=True)
        self.logger.info("INFO: Using Propensity Score Stratification Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

        self.num_strata = num_strata
        self.clipping_threshold = clipping_threshold

    def _estimate_effect(self):
        propensity_score_model = linear_model.LinearRegression()
        propensity_score_model.fit(self._observed_common_causes, self._treatment)
        self._data['propensity_score'] = propensity_score_model.predict(self._observed_common_causes)

        # sort the dataframe by propensity score
        # create a column 'strata' for each element that marks what strata it belongs to
        num_rows = self._data[self._outcome_name].shape[0]
        self._data['strata'] = (
            (self._data['propensity_score'].rank(ascending=True) / num_rows) * self.num_strata
        ).round(0)

        # for each strata, count how many treated and control units there are
        # throw away strata that have insufficient treatment or control
        #print("before clipping, here is the distribution of treatment and control per strata")
        #print(self._data.groupby(['strata',self._treatment_name])[self._outcome_name].count())

        # convert lists of single elements to strs
        if isinstance(self._treatment_name, list) and len(self._treatment_name)==1:
            self._treatment_name=self._treatment_name[0]

        if isinstance(self._outcome_name, list) and len(self._outcome_name)==1:
            self._outcome_name=self._outcome_name[0]

        # calcs
        self._data['dbar'] = 1 - self._data[self._treatment_name]
        self._data['d_y'] = self._data[self._treatment_name] * self._data[self._outcome_name]
        self._data['dbar_y'] = self._data['dbar'] * self._data[self._outcome_name]

        stratified = self._data.groupby('strata')
        clipped = stratified.filter(
            lambda strata: min(strata.loc[strata[strata[self._treatment_name] == 1].index].shape[0],
                               strata.loc[strata[strata[self._treatment_name] == 0].index].shape[0]) > self.clipping_threshold)

        # sum weighted outcomes over all strata  (weight by treated population) 
        weighted_outcomes = clipped.groupby('strata').agg({self._treatment_name: np.sum, 'dbar': np.sum, 'd_y': np.sum,'dbar_y': np.sum})
        weighted_outcomes.columns = [x+"_sum" for x in weighted_outcomes.columns]
        weighted_outcomes.to_csv("weightedoutcomes.csv")
        treatment_sum_name = self._treatment_name + "_sum"

        weighted_outcomes['d_y_mean'] = weighted_outcomes['d_y_sum'] / weighted_outcomes[treatment_sum_name]
        weighted_outcomes['dbar_y_mean'] = weighted_outcomes['dbar_y_sum'] / weighted_outcomes['dbar_sum']
        weighted_outcomes['effect'] = weighted_outcomes['d_y_mean'] - weighted_outcomes['dbar_y_mean']
        total_treatment_population = weighted_outcomes[treatment_sum_name].sum()

        ate = (weighted_outcomes['effect'] * weighted_outcomes[treatment_sum_name]).sum() / total_treatment_population
        # TODO - how can we add additional information into the returned estimate?
        #        such as how much clipping was done, or per-strata info for debugging?
        estimate = CausalEstimate(estimate=ate,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator)
        return estimate

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        var_list = estimand.treatment_variable + estimand.backdoor_variables
        expr += "+".join(var_list)
        return expr
