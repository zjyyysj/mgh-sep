


num = 1073;
[obs_num, feature_num] = size(patient_path);
[obs_num1, feature_num1] = size(patient_process1);
[obs_num2, ~] = size(patient_process2);
patient_exist = zeros(num,1);
patient = zeros(1000, feature_num); 
patient_process = zeros(1000, feature_num1);
patient80 = zeros(100,1);
patient_exist_index = zeros(600,1);

index = 1;
num80 = 0; 
for i = 1:num
    index_s = 0;
    
    for j = 1:obs_num
        
        if test_set{i}.icuid - 200000 == patient_path(j, 2)
            patient(index,:) = patient_path(j,:);
            index = index +1;
            index_s = index_s + 1;
        end
        
    end
    patient_exist(i) = index_s;
    if index_s == 80
        num80 = num80 +1;
        patient80(num80) = i;
    end
end

index = 1;
for i = 1:num   
    for j = 1:obs_num1
        
        if test_set{i}.icuid - 200000 == patient_process1(j, 2)
            patient_process(index,:) = patient_process1(j,:);
            index = index +1;
        end
    end
        
     for j = 1:obs_num2
        if test_set{i}.icuid - 200000 == patient_process2(j, 2)
            patient_process(index,:) = patient_process2(j,:);
            index = index +1;
        end
     end
end

writematrix(patient_process, 'patient_individual.csv', 'Delimiter', ',');
writematrix(patient, 'patient_individual_discrete.csv', 'Delimiter', ',');

index = 1;
for i =1:num
    if patient_exist(i) > 0
        patient_exist_index(index) = i;
        index = index+1;
    end
end

save patient_exist_index patient_exist_index patient_exist patient_process