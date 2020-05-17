load('models.mat');

model_value_comparison(models)
best_model = get_best_model(models);
best_model = add_normalized_data_to_model(best_model);
best_model = add_policy_information_to_model(best_model);

test_set = best_model.test_set;
save test_set test_set;

