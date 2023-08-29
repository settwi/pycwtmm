from dataclasses import dataclass
import json
import os

import numpy as np
import scipy.stats as st


@dataclass
class SlurmCwtmmDataPreprocessor:
    energy_bins: np.ndarray
    counts: np.ndarray
    dir_prefix: str
    num_threads: int
    num_draws: int
    script_to_run: str

    @property
    def json_dir(self):
        return self.dir_prefix + '-json'

    @property
    def jobs_dir(self):
        return self.dir_prefix + '-jobs'

    @property
    def data_out_dir(self):
        return self.dir_prefix + '-data-out'

    def build_directories(self):
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.jobs_dir, exist_ok=True)
        os.makedirs(self.data_out_dir, exist_ok=True)

    def write_tons_of_json(self):
        for band_idx, (eb, y) in enumerate(zip(self.energy_bins, self.counts)):
            for draw_idx in range(self.num_draws):
                if isinstance(y, int): draw = st.poisson.rvs(y)
                else: draw = st.norm.rvs(loc=y, scale=np.sqrt(np.abs(y)))
                save = {
                    'energy_band': list(eb.astype(float)),
                    'counts': list(draw)
                }

                out_fn = f'{self.json_dir}/{draw_idx}-{band_idx}.json'
                with open(out_fn, 'w') as f:
                    f.write(json.dumps(save))

    def write_slurms(self):
        SLURM_HEAD = ''\
            '#!/bin/bash -l\n'\
            '#SBATCH --time=1:00:00\n'\
            '#SBATCH --ntasks={:d}\n'\
            '#SBATCH --mem=20g\n'\
            '#SBATCH --tmp=10g\n'\
            '#SBATCH --mail-type=NONE\n'\
            '#SBATCH --mail-user=sette095@umn.edu\n'

        fnames = os.listdir(self.json_dir)
        num_jobs = int(np.ceil(len(fnames) / self.num_threads))
        head = SLURM_HEAD.format(self.num_threads)

        for job_idx in range(num_jobs):
            with open(f'{self.jobs_dir}/job{job_idx}.txt', 'w') as f:
                print(head, file=f)
                print(file=f)

                start = job_idx * self.num_threads
                end = start + self.num_threads

                json_files = ' '.join(
                    f"'../{self.json_dir}/{fn}'" for fn in fnames[start:end]
                )
                cmd = f"python '{self.script_to_run}' "
                cmd += f"--job_name 'job{job_idx}' "
                cmd += f"--num_threads '{self.num_threads}' "
                cmd += f"--out_dir '../{self.data_out_dir}' "
                cmd += "--json_files " + json_files
                print(cmd, file=f)

