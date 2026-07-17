GRAPHS commands
>> python plot_fl_loss_acc.py --max_round 4000 --ema 0.05   --out_prefix Finalplots_CIFAR100_   --algo FedExpSls=/home/somya/rungeetika/FederatedLineSearch/fedexpsls_23feb_cifar100_seed0.log --algo FedAvg=/home/somya/rungeetika/FederatedLineSearch/fedavg_CIFAR100_4000_seed0.log   --algo FedExp=/home/somya/rungeetika/FederatedLineSearch/fedexp_CIFAR100_4000_seed0.log   --algo FedExProx=/home/somya/rungeetika/FederatedLineSearch/fedproxexp_CIFAR100_seed0_clean.log   --algo FedSls=/home/somya/rungeetika/FederatedLineSearch/fedsls_23feb_cifar100_seed0.log   --algo FedAdam=/home/somya/rungeetika/FederatedLineSearch/fedadam_CIFAR100_4000_seed0.log --pdf

>>     python plot_fl_loss_acc.py --max_round 4000 --ema 0.05   --out_prefix Finalplots_FEMNIST_   --algo FedExpSls=/home/somya/rungeetika/FederatedLineSearch/fedexpsls_23feb_femnist_seed0.log --algo FedAvg=/home/somya/rungeetika/FederatedLineSearch/fedavg_femnist_seed0.log   --algo FedExp=/home/somya/rungeetika/FederatedLineSearch/fedexp_femnist_seed0.log   --algo FedExProx=/home/somya/rungeetika/FederatedLineSearch/fedexprox_femnist_seed0.log   --algo FedSls=/home/somya/rungeetika/FederatedLineSearch/fedsls_23feb_femnist_seed0.log   --algo FedAdam=/home/somya/rungeetika/FederatedLineSearch/fedadam_femnist_seed0.log --pdf

> run the command for the experiments:
 CUDA_VISIBLE_DEVICES=3  nohup python main.py   --seed 3   --algorithm "fedexpsls"   --dataset shakespeare   --model LSTM   --num_clients 100   --num_participating_clients 20   --num_rounds 500   --alpha 0.3   >logs/fedexpsls_shakespeare_seed3.log 2>&1 &

 >> for csvs:
 for s in 0 1 2 3 4; do   python plot_fl_loss_acc.py     --algo "FedExpSls=logs/shakespeare_results/fedexpsls_shakespeare_seed${s}.log"    
 --max_round 500     --out_prefix "csv/FedExpSls_seed${s}_"     --dump_csv; done

 >> for avg_csv:
 python avg_seeds_metrics.py \
  --pattern "csv/FedAvg_seed*_FedAvg_train_loss.csv" \
  --out "avg_csv/FedAvg_train_loss_avg.csv"

python avg_seeds_metrics.py \
  --pattern "csv/FedAvg_seed*_FedAvg_test_acc.csv" \
  --out "avg_csv/FedAvg_test_acc_avg.csv"

  python avg_seeds_metrics.py \
  --pattern "csv/FedExProx_seed*_FedExProx_train_loss.csv" \
  --out "avg_csv/FedExProx_train_loss_avg.csv"

python avg_seeds_metrics.py \
  --pattern "csv/FedExProx_seed*_FedExProx_test_acc.csv" \
  --out "avg_csv/FedExProx_test_acc_avg.csv"

>> for train plots averaged:
python plot_avg_rounds.py \
  --algo "FedExpSls=avg_csv/FedExpSls_train_loss_avg.csv" \
  --algo "FedAvg=avg_csv/FedAvg_train_loss_avg.csv" \
  --algo "FedExp=avg_csv/FedExp_train_loss_avg.csv" \
  --algo "FedExpProx=avg_csv/FedExProx_train_loss_avg.csv" \
  --algo "FedSls=avg_csv/FedSls_train_loss_avg.csv" \
  --algo "FedAdam=avg_csv/FedAdam_train_loss_avg.csv" \
  --out figs/train_loss_avg.png \
  --pdf --band

  >> for test plots averaged:
  python plot_avg_rounds.py \
  --algo "FedExpSls=avg_csv/FedExpSls_test_acc_avg.csv" \
  --algo "FedAvg=avg_csv/FedAvg_test_acc_avg.csv" \
  --algo "FedExp=avg_csv/FedExp_test_acc_avg.csv" \
  --algo "FedExpProx=avg_csv/FedExProx_test_acc_avg.csv" \
  --algo "FedSls=avg_csv/FedSls_test_acc_avg.csv" \
  --algo "FedAdam=avg_csv/FedAdam_test_acc_avg.csv" \
  --out figs/test_acc_avg.png \
  --pdf --band

  python avg_wallclock.py   --pattern "wallclock_csv/FedSls_seed*_time_train_loss.csv"   --out "avg_wallclock_csv/FedSls_time_train_loss_avg
.csv"   --dt 60   --min_seeds 5

    python avg_wallclock.py   --pattern "wallclock_csv/FedAvg_seed*_time_test_acc.csv"   --out "avg_wallclock_csv/FedAvg_time_test_acc_avg.csv"   --dt 60   --min_seeds 5

    python plot_avg_time.py \
  --algo "FedAvg=avg_wallclock_csv/FedAvg_time_train_loss_avg.csv" \
  --algo "FedExpSls=avg_wallclock_csv/FedExpSls_time_train_loss_avg.csv" \
  --algo "FedExp=avg_wallclock_csv/FedExp_time_train_loss_avg.csv" \
  --algo "FedExpProx=avg_wallclock_csv/FedExProx_time_train_loss_avg.csv" \
  --algo "FedSls=avg_wallclock_csv/FedSls_time_train_loss_avg.csv" \
  --algo "FedAdam=avg_wallclock_csv/FedAdam_time_train_loss_avg.csv" \
  --out figs/train_loss_vs_time_avg.png \
  --pdf

  python plot_avg_time.py \
  --algo "FedAvg=avg_wallclock_csv/FedAvg_time_test_acc_avg.csv" \
  --algo "FedExpSls=avg_wallclock_csv/FedExpSls_time_test_acc_avg.csv" \
  --algo "FedExp=avg_wallclock_csv/FedExp_time_test_acc_avg.csv" \
  --algo "FedExpProx=avg_wallclock_csv/FedExProx_time_test_acc_avg.csv" \
  --algo "FedSls=avg_wallclock_csv/FedSls_time_test_acc_avg.csv" \
  --algo "FedAdam=avg_wallclock_csv/FedAdam_time_test_acc_avg.csv" \
  --out figs/test_acc_vs_time_avg.png \
  --pdf

python plot_avg_time.py \
  --algo "FedAvg=wallclock_csv/FedAvg_seed0c100time_train_loss.csv" \
  --algo "FedExpSls=wallclock_csv/FedExpSls_seed0c100_time_train_loss.csv" \
  --algo "FedExp=wallclock_csv/FedExp_seed0c100_time_train_loss.csv" \
  --algo "FedExpProx=wallclock_csv/FedExProx_seed0c100_time_train_loss.csv" \
  --algo "FedSls=wallclock_csv/FedSls_seed0c100time_train_loss.csv" \
  --algo "FedAdam=wallclock_csv/FedAdam_seed0c100_time_train_loss.csv" \
  --out figs/cifar100_train_loss_vs_time_avg.png \
  --pdf

 python plot_avg_time.py \
  --algo "FedAvg=avg_wallclock_csv/FedAvg_time_test_acc_avg.csv" \
  --algo "FedExpSls=avg_wallclock_csv/FedExpSls_time_test_acc_avg.csv" \
  --algo "FedExp=avg_wallclock_csv/FedExp_time_test_acc_avg.csv" \
  --algo "FedExpProx=avg_wallclock_csv/FedExProx_time_test_acc_avg.csv" \
  --algo "FedSls=avg_wallclock_csv/FedSls_time_test_acc_avg.csv" \
  --algo "FedAdam=avg_wallclock_csv/FedAdam_time_test_acc_avg.csv" \
  --ylabel "Test Accuracy (%)" \
  --title "Test Accuracy vs Time" \
  --out acc_time.png \
  --pdf


python plot_avg_time.py \
  --algo "FedAvg=avg_wallclock_csv/FedAvg_time_train_loss_avg.csv" \
  --algo "FedExpSls=avg_wallclock_csv/FedExpSls_time_train_loss_avg.csv" \
  --algo "FedExp=avg_wallclock_csv/FedExp_time_train_loss_avg.csv" \
  --algo "FedExpProx=avg_wallclock_csv/FedExProx_time_train_loss_avg.csv" \
  --algo "FedSls=avg_wallclock_csv/FedSls_time_train_loss_avg.csv" \
  --algo "FedAdam=avg_wallclock_csv/FedAdam_time_train_loss_avg.csv" \
  --ylabel "Train Loss" \
  --title "Train Loss vs Time" \
  --out loss_time.png \
  --pdf