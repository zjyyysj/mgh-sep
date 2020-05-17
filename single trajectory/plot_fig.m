
real_clinician = true;
load('patient_exist_info.mat')
load('test_set.mat')
load('target_action.mat')
 
N = length(patient_exist_index);
median_iv_dose = [0,30,85,320,946];
median_vaso_dose = [0,0.04,0.13,0.27,0.68];

index = 1;
for i = 1 : N
    
    patient   = test_set{patient_exist_index(i)};
    num_steps = length(patient.discrete_trajectory);
    d = double(patient.mortality);
    
    if real_clinician
        continuous_actions_clinician = patient.cont_actions';
    end
    
    continuous_actions_ai = patient.continuous_actions_ai;
    
    MAP = patient.MAP;
    MAPIndex = 1:length(MAP);
    
    
    test_set{patient_exist_index(i)} = patient;
    
    num_steps_exist = patient_exist(patient_exist_index(i));
    patient_bloc = patient_process(index:(index+num_steps_exist-1),1);
    patient_action = target_action(index:(index+num_steps_exist-1),:);
    num_bloc_exist = patient_bloc(num_steps_exist);
    patient_action_actual = zeros(num_bloc_exist,2);
    for j = 1:(num_steps_exist)
        dif =0;
        patient_action_actual(patient_bloc(j),1) = median_iv_dose(patient_action(j,3)+1);
        patient_action_actual(patient_bloc(j),2) = median_vaso_dose(patient_action(j,4)+1);
        if (patient_bloc(j) - j) > dif
            for k=1: (patient_bloc(j) - j - dif)
                patient_action_actual(patient_bloc(j)-k,1) = median_iv_dose(patient_action(j-1,3)+1);
                patient_action_actual(patient_bloc(j)-k,2) = median_vaso_dose(patient_action(j-1,4)+1);
            end
            dif = dif + patient_bloc(j) - j;
        end
    end
             
    index = index + num_steps_exist;
    
    close all;
    
    navy   = [0.071, 0.184, 0.239];
    red    = [0.745, 0.243, 0.169];
    green  = [0.05, 0.75, 0.25];
    
    Nlen = length(continuous_actions_clinician(:, 1));
    f = figure('visible', 'off');
    
    set(f, 'Position', [0, 0, 1000, 800])
    set(gca, 'fontsize', 30)
    
    ax(1)=subplot(3,1,1);
    h = stem(continuous_actions_ai(:, 1),'-.', 'Filled');
    h(1).LineWidth = 2;
    h(1).MarkerSize = 8;
    h(1).Color = navy;
    ylim([0 110]);
    hbase = h.BaseLine;
    hbase.LineStyle = '--';
    hbase.LineWidth = 2;
    hold on,
    h = stem(continuous_actions_clinician(:, 1),'-.','color', red);
    h(1).LineWidth = 2;
    h(1).MarkerSize = 8;
    ylim([0 110]);
    hbase = h.BaseLine;
    hbase.LineStyle = '--';
    hbase.LineWidth = 2;
    hold on,
    h = stem(patient_action_actual(:, 2),'-.','color', green);
    h(1).LineWidth = 2;
    h(1).MarkerSize = 8;
    ylim([0 110]);
    hbase = h.BaseLine;
    hbase.LineStyle = '--';
    hbase.LineWidth = 2;
    ylim([0 0.75]);
    xlim([0-0.5 Nlen+0.5])
    ylabel('Pressors');
    hleg = legend('RL Agent','Clinician','DRL Agent','orientation','horizontal', 'location', 'northoutside');
    set(gca,'FontSize',25);
    set(gca,'xticklabel',[])
    
    ax(2) =subplot(3,1,2);
    h = stem(continuous_actions_ai(:, 2),'--', 'Filled');
    h(1).LineWidth = 2;
    h(1).MarkerSize = 8;
    h(1).Color = navy;
    hold on,
    h = stem(continuous_actions_clinician(:, 2),'--','color', red);
    h(1).LineWidth = 2;
    h(1).MarkerSize = 8;
    ylim([0 110]);
    hbase = h.BaseLine;
    hbase.LineStyle = '--';
    hbase.LineWidth = 2;
    hold on,
    h = stem(patient_action_actual(:, 1),'-.','color', green);
    h(1).LineWidth = 2;
    h(1).MarkerSize = 8;
    ylim([0 110]);
    hbase = h.BaseLine;
    hbase.LineStyle = '--';
    hbase.LineWidth = 2;
    ylim([0 1500]);
    xlim([0-0.5 Nlen+0.5]),
    ylabel('Fluids');
    xlabel('Time (Hours)')
    set(gca, 'fontsize', 25)
    
    ax(3)=subplot(3,1,3);
    
    h = stem(MAPIndex(MAP > 65), MAP(MAP > 65), 'Filled', 'BaseValue', 65);
    h(1).LineWidth = 2;
    h(1).MarkerSize = 8;
    h(1).Color = navy;
    hold on
    if ~isempty(MAPIndex(MAP < 65))
        h = stem(MAPIndex(MAP < 65), MAP(MAP < 65), 'Filled', 'BaseValue', 65);
        h(1).LineWidth = 2;
        h(1).MarkerSize = 8;
        h(1).Color = red;
    end
    hbase = h.BaseLine;
    hbase.LineStyle = '--';
    hbase.LineWidth = 2;
    ylabel('MAP');ylim([40 110]);
    xlim([0-0.5 Nlen+0.5]);
    set(gca,'FontSize',25),
    xlabel('Time (Hours)')
    
    print(['Figures_noise/' num2str(i) '_actions_noise' num2str(d)], '-dpng')
    close;
end
