import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. سحب البيانات
df = pd.read_csv('medical_examination.csv')

# 2. إضافة عمود overweight
# بنحسب الـ BMI ونشوف لو أكبر من 25 ندي 1، ولو لأ ندي 0
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3. توحيد البيانات (Normalization)
# 0 للطبيعي (1) و 1 لأي حاجة أعلى من الطبيعي
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. دالة رسم الـ Categorical Plot
def draw_cat_plot():
    # 5. تجهيز البيانات بالـ Melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. تجميع البيانات للعد (Groupby)
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. رسم الـ Catplot باستخدام Seaborn
    fig = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar').fig

    # 8. حفظ الصورة
    fig.savefig('catplot.png')
    return fig

# 10. دالة رسم الـ Heat Map
def draw_heat_map():
    # 11. تنظيف البيانات (حذف القيم الشاذة Outliers)
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. حساب معامل الارتباط
    corr = df_heat.corr()

    # 13. عمل ماسك للمثلث العلوي عشان الرسمة تبقى أوضح
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. إعداد الـ Figure بتاع Matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. رسم الـ Heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, square=True, linewidths=.5, cbar_kws={'shrink': .5})

    # 16. حفظ الصورة
    fig.savefig('heatmap.png')
    return fig