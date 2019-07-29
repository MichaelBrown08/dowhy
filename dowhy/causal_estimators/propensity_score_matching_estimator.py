from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
import pandas as pd

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimator import CausalEstimator


class PropensityScoreMatchingEstimator(CausalEstimator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.debug("Back-door variables used:" +
                          ",".join(self._target_estimand.backdoor_variables))
        self._observed_common_causes_names = self._target_estimand.backdoor_variables
        self._observed_common_causes = self._data[self._observed_common_causes_names]
        self._observed_common_causes = pd.get_dummies(self._observed_common_causes, drop_first=True)
        self.logger.info("INFO: Using Propensity Score Matching Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

    def _estimate_effect(self):
        propensity_score_model = linear_model.LinearRegression()
        propensity_score_model.fit(self._observed_common_causes, self._treatment)
        self._data['propensity_score'] = propensity_score_model.predict(self._observed_common_causes)

        self._data.to_csv("base_data.csv")

        # this assumes a binary treatment regime
        treated = self._data.loc[self._data[self._treatment_name]==1]
        control = self._data.loc[self._data[self._treatment_name]==0]

        treated.to_csv("base_treated.csv")
        control.to_csv("base_control.csv")

        control_neighbors = (
            NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
            .fit(control['propensity_score'].values.reshape(-1, 1))
        )
        distances, indices = control_neighbors.kneighbors(treated['propensity_score'].values.reshape(-1, 1))

        # TODO remove neighbors that are more than a given radius apart

        # estimate ATE on treated by summing over difference between matched neighbors
        ate = 0
        numtreatedunits = control.shape[0]
        print(f"Numtreatedunits: {numtreatedunits}; len(indices): {len(indices)}.")
        assert len(indices)==numtreatedunits
        for i in range(numtreatedunits):
            treated_outcome = treated.iloc[i][self._outcome_name].item()
            try:
                control_outcome = control.iloc[indices[i]][self._outcome_name].item()
            except AttributeError:
                print(f"Attribute error raised at {i}")
                indices=pd.DataFrame(indices)
                indices.to_csv("breakpointindices.csv")
                control.to_csv("breakpoint_control.csv")
                treated.to_csv("breakpoint_treated.csv")
                break
            print(f"Loop {i}; indices {indices[i]}, to: {treated_outcome}, co: {control_outcome}")
            ate += treated_outcome - control_outcome

        ate /= numtreatedunits
        estimate = CausalEstimate(estimate=ate,
                                  target_estimand=self._target_estimand,
                                  realized_estimand_expr=self.symbolic_estimator)
        return estimate

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ", ".join(estimand.outcome_variable) + "~"
        # TODO -- fix: we are actually conditioning on positive treatment (d=1)
        var_list = estimand.treatment_variable + estimand.backdoor_variables
        expr += "+".join(var_list)
        return expr
