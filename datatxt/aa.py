import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
df = pd.read_csv('kdp_abs_diff_wide.csv')

# 2. 筛选 8-15km 范围数据
mask = (df['dis_mid'] >= 8) & (df['dis_mid'] <= 15)
filtered_df = df[mask].copy()

# 3. 准备绘图数据
# 这里直接将列名重命名为您要求的 LaTeX 格式：Raw-KDP 和 Reg-KDP
rename_dict = {
    'raw_kdp_abs_diff': r'$Raw-K_{DP}$',
    'kdp_lsf_abs_diff': r'$Reg-K_{DP}$'
}

# 仅保留需要的两列并重命名
plot_df = filtered_df[['raw_kdp_abs_diff', 'kdp_lsf_abs_diff']].rename(columns=rename_dict)

# 4. 转换数据格式 (Melt)
# 关键修复：value_name 使用简单的 'Value'，避免空格导致的 ValueError
melted_df = plot_df.melt(var_name='Method', value_name='Value')

# 5. 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix' # 使数学公式风格与 Times 一致

# 6. 开始绘图
plt.figure(figsize=(8, 6))

# 绘图时 y='Value' 必须与 melt 里的 value_name 一致
sns.boxplot(
    x='Method',
    y='Value',
    data=melted_df,
    palette="Set2",
    width=0.5
)

# 7. 设置纯英文坐标轴标题
plt.xlabel('Method', fontsize=12)
plt.ylabel('Absolute Difference', fontsize=12)

# 移除标题 (根据您的要求)
plt.title('')

plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# 保存并显示
plt.savefig('boxplot_fixed_english.png', dpi=300)
plt.show()