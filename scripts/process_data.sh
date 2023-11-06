re_submit=(150)
for i in "${re_submit[@]}" #`seq 101 200`
do
echo "#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=110G
#####SBATCH --exclusive
#####SBATCH --gres=gpu:2
#####SBATCH --partition=gpu
#####SBATCH --partition xlong 

##OPTIONAL JOB SPECIFICATIONS
####SBATCH --mail-type=ALL
####SBATCH --mail-user=sunyuanfei@tamu.edu
#SBATCH --account=122807222934
#SBATCH --job-name=dt_pro_${i}
#SBATCH --output=/scratch/user/sunyuanfei/Projects/M3_PLM/logs/process_pdb_dataset_${i}.out


#First Executable Line

module purge
source ~/.bashrc
#module load WebProxy
#module load cuDNN/8.0.5.39-CUDA-11.1.1 GCC/11.3.0
#module load cuDNN/8.0.4.30-CUDA-11.1.1 GCC/11.3.0
#module load cuDNN/8.2.1.32-CUDA-11.3.1 GCC/11.3.0
conda activate env_lightning

python src/data/components/structure_utils/process_pdb_dataset.py --mmcif_dir=/scratch/user/sunyuanfei/Projects/OpenProteinSet/pdb_mmcif_subs/dir_${i} --min_file_size=1000 --max_resolution=5.0 --max_len=5000 --write_dir=/scratch/user/sunyuanfei/Projects/M3_PLM/data/pdb_pickles --tmp_dir=/scratch/user/sunyuanfei/Projects/M3_PLM/data/tmp_dir --PTGL_path=/scratch/user/sunyuanfei/Projects/PTGLtools/PTGLgraphComputation --OpenProtein_path=/scratch/user/sunyuanfei/Projects/OpenProteinSet --verbose
" > scripts/process_data.job
sbatch scripts/process_data.job
done
