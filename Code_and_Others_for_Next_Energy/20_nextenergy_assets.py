"""Step 20 - Next Energy figure files (run after 07, 08 and 12).
Copies the figures, which steps 07, 08 and 12 save under the article numbering, to figures/NextEnergy:
Fig. 1 study design, Fig. 2 unseen-domain error, Fig. 3 certificate, Fig. 4 simulation with known truth,
Fig. 5 conformal bounds, Fig. 6 reference checks, Fig. 7 explainability, Fig. 8 generalizability,
Fig. S1 reliability (Supplementary Material 1). The graphical abstract is drawn by step 21."""
import os, shutil

FIG = 'figures'; NE = os.path.join(FIG, 'NextEnergy'); os.makedirs(NE, exist_ok=True)
for n in ['Figure_1', 'Figure_2', 'Figure_3', 'Figure_4', 'Figure_5', 'Figure_6', 'Figure_7', 'Figure_8', 'Figure_S1']:
    shutil.copy(os.path.join(FIG, n + '.png'), os.path.join(NE, n + '.png'))
print('Next Energy figures written to', NE)
