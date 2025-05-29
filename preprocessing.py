import pandas as pd

# Baca dataset dengan header bahasa Indonesia
df = pd.read_csv('data/diabetes.csv', delimiter=';')

# Pastikan kolom tanpa nama tidak ikut terbaca (drop otomatis jika ada)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Ambil hanya kolom yang diperlukan
kolom_diambil = ['Insulin', 'Glukosa', 'Fungsi Keturunan Diabetes', 'Hasil']
df_selected = df[kolom_diambil]

# Simpan ke file baru (opsional, bisa juga hanya print)
df_selected.to_csv('data/diabetes_selected.csv', index=False, sep=';')

# Tampilkan hasil
print(df_selected.head())
