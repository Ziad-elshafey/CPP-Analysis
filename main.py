import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import numpy as np


def extract_numbers_from_fee(fee_str):
    """Return a list of numeric amounts found in the fee string.

    Supports formats like:
    - "$10,000"
    - "10000"
    - "25k" or "25K"
    - "Gold=25000,Black=15000"
    Returns floats (USD)."""
    if not isinstance(fee_str, str):
        return []
    s = fee_str.replace('$', '')
    # Find numbers with optional commas, decimals, optional trailing k/K
    parts = re.findall(r"(\d+(?:,\d{3})*(?:\.\d+)?\s*[kK]?)", s)
    nums = []
    for p in parts:
        p_clean = p.replace(',', '').strip()
        if p_clean.lower().endswith('k'):
            try:
                nums.append(float(p_clean[:-1]) * 1000.0)
            except Exception:
                continue
        else:
            try:
                nums.append(float(p_clean))
            except Exception:
                continue
    return nums


def calculate_average_fee(fee_str):
    nums = extract_numbers_from_fee(fee_str)
    return float(np.mean(nums)) if nums else None


def analyze_data(file_path='gathered_data.csv'):
    """
    Analyzes the corporate partnership data and generates visualizations.
    """
    # Create a directory to save plots
    if not os.path.exists('plots'):
        os.makedirs('plots')

    df = pd.read_csv(file_path)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    # --- Analysis 1: Number of Public Partners ---
    print("Generating partner count visualization...")
    plt.figure(figsize=(14, 10))
    partners_df = df[df['No Of Public Partners'].notna()].sort_values(
        by='No Of Public Partners', ascending=False)
    colors = sns.color_palette("viridis", len(partners_df))
    ax = sns.barplot(x='No Of Public Partners', y='Institution',
                     data=partners_df, palette=colors)
    plt.title('Number of Public Corporate Partners per Institution',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Public Partners', fontsize=12)
    plt.ylabel('Institution', fontsize=12)
    
    # Add value labels
    for i, v in enumerate(partners_df['No Of Public Partners']):
        ax.text(v + 1, i, str(int(v)), va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/partners_per_institution.png', bbox_inches='tight')
    plt.close()

    # --- Analysis 2: Technology Sectors Represented ---
    print("Generating sector analysis visualization...")
    # Clean sectors: split, strip and remove 'Not posted' / 'Not listed' artifacts
    def clean_sector_field(s):
        if not isinstance(s, str):
            return []
        items = [it.strip() for it in s.split(',')]
        cleaned = []
        for it in items:
            it_low = it.lower()
            if not it or it_low in ('not listed', 'not posted', 'not post', 'not'):
                continue
            # ignore tokens that are clearly placeholders
            if re.search(r'not\s*(posted|listed)', it_low):
                continue
            cleaned.append(it)
        return cleaned

    sectors_series = df['Sectors Represented'].dropna().apply(clean_sector_field)
    sectors = sectors_series.explode().dropna()
    sectors = sectors.str.strip()
    sector_counts = sectors.value_counts()

    plt.figure(figsize=(14, 10))
    top_sectors = sector_counts.head(15)
    colors = sns.color_palette("Blues_r", len(top_sectors))
    ax = top_sectors.sort_values(ascending=True).plot(kind='barh',
                                                       color=colors)
    plt.title('Top 15 Most Represented Technology Sectors',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Frequency (Number of Institutions)', fontsize=12)
    plt.ylabel('Sector', fontsize=12)
    
    # Add value labels
    for i, v in enumerate(top_sectors.sort_values(ascending=True)):
        ax.text(v + 0.2, i, str(v), va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/top_sectors.png', bbox_inches='tight')
    plt.close()

    # --- Analysis 3: Membership Fees (Average) ---
    print("Generating membership fee analysis...")
    # Consider a program as fee-public if the 'Fees Public' field suggests yes/fee/partial/variable
    fees_flag = df['Fees Public'].astype(str).str.lower().fillna('')
    fee_public_mask = fees_flag.str.contains(r'yes|fee|variable|partial')
    fee_df_all = df[fee_public_mask].copy()
    fee_df_all['Fee Mean'] = fee_df_all['Fee Notes'].apply(calculate_average_fee)
    # Save processed fee info for later inspection
    # Institutions with Fee Mean == NaN will be shown as 'Amount not posted'

    # Separate those with numeric fee means vs those without
    fee_with_amounts = fee_df_all[fee_df_all['Fee Mean'].notna()].sort_values('Fee Mean')
    fee_no_amount = fee_df_all[fee_df_all['Fee Mean'].isna()]

    plt.figure(figsize=(14, max(6, len(fee_with_amounts) * 0.5)))
    if not fee_with_amounts.empty:
        colors = sns.color_palette("rocket", len(fee_with_amounts))
        ax = plt.barh(fee_with_amounts['Institution'], fee_with_amounts['Fee Mean'], color=colors)
        plt.xlabel('Average Fee Amount (USD)', fontsize=12)
        plt.title('Average Membership Fee by Institution (Where Public and Amount Posted)',
                  fontsize=16, fontweight='bold', pad=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        # Add labels
        for i, fee in enumerate(fee_with_amounts['Fee Mean']):
            plt.text(fee + max(500, fee * 0.02), i, f'${fee:,.0f}', va='center', fontsize=9)
    else:
        plt.text(0.5, 0.5, 'No public fee amounts found', ha='center')

    # Show institutions that declare fees but do not list amounts as markers below the chart
    if not fee_no_amount.empty:
        # Place them in a separate small panel below via annotation listing
        y_text = -1
        plt.subplots_adjust(bottom=0.25)
        note = 'Programs listing fees but not posting amounts: ' + ', '.join(fee_no_amount['Institution'].tolist())
        plt.figtext(0.01, 0.02, note, wrap=True, horizontalalignment='left', fontsize=9)

    plt.tight_layout()
    plt.savefig('plots/average_membership_fees.png', bbox_inches='tight')
    plt.close()

    # --- Analysis 4: Tiers System ---
    print("Generating tier system analysis...")
    plt.figure(figsize=(10, 8))
    tier_counts = df['Tiers'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4']
    wedges, texts, autotexts = plt.pie(tier_counts, labels=tier_counts.index,
                                        autopct='%1.1f%%', colors=colors,
                                        startangle=90, textprops={'fontsize': 12})
    plt.title('Proportion of Programs with Tiered Membership',
              fontsize=16, fontweight='bold', pad=20)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    plt.savefig('plots/tiered_vs_notier.png', bbox_inches='tight')
    plt.close()

    # --- Analysis 5: Program Scope Distribution ---
    print("Generating program scope analysis...")
    plt.figure(figsize=(12, 8))
    scope_counts = df['Program Scope'].value_counts()
    colors = sns.color_palette("Set2", len(scope_counts))
    plt.pie(scope_counts, labels=scope_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=45)
    plt.title('Distribution of Program Scope (Department vs School/College)',
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig('plots/program_scope.png', bbox_inches='tight')
    plt.close()

    # --- Analysis 6: Benefits Offered ---
    print("Generating benefits analysis...")
    benefits = df['Benefits'].dropna().str.split(',').explode()
    benefit_counts = benefits.str.strip().value_counts().head(10)

    plt.figure(figsize=(14, 9))
    colors = sns.color_palette("magma", len(benefit_counts))
    ax = benefit_counts.sort_values(ascending=True).plot(kind='barh',
                                                          color=colors)
    plt.title('Top 10 Most Common Benefits Offered to Partners',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Frequency (Number of Institutions)', fontsize=12)
    plt.ylabel('Benefit', fontsize=12)
    
    # Add value labels
    for i, v in enumerate(benefit_counts.sort_values(ascending=True)):
        ax.text(v + 0.2, i, str(v), va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/top_benefits.png', bbox_inches='tight')
    plt.close()

    # --- Analysis 7: Top Corporate Partners Network ---
    print("Generating top partners analysis...")
    all_partners = []
    for partners_str in df['Top Partners'].dropna():
        partners_list = [p.strip() for p in partners_str.split(',')]
        all_partners.extend(partners_list)
    
    partner_counts = pd.Series(all_partners).value_counts().head(15)

    plt.figure(figsize=(14, 10))
    colors = sns.color_palette("coolwarm", len(partner_counts))
    ax = partner_counts.sort_values(ascending=True).plot(kind='barh',
                                                          color=colors)
    plt.title('Top 15 Most Frequently Appearing Corporate Partners',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Partnerships', fontsize=12)
    plt.ylabel('Company', fontsize=12)
    
    # Add value labels
    for i, v in enumerate(partner_counts.sort_values(ascending=True)):
        ax.text(v + 0.1, i, str(v), va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/top_corporate_partners.png', bbox_inches='tight')
    plt.close()

    # --- Analysis 8: Fee Transparency ---
    print("Generating fee transparency analysis...")
    plt.figure(figsize=(10, 8))
    fee_public_counts = df['Fees Public'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    wedges, texts, autotexts = plt.pie(fee_public_counts,
                                        labels=fee_public_counts.index,
                                        autopct='%1.1f%%', colors=colors,
                                        startangle=90)
    plt.title('Fee Information Transparency Across Programs',
              fontsize=16, fontweight='bold', pad=20)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    plt.savefig('plots/fee_transparency.png', bbox_inches='tight')
    plt.close()

    # --- Analysis 9: Transparency Score ---
    print("Generating transparency score analysis...")

    # Calculate transparency score (0-100)
    def calculate_transparency_score(row):
        score = 0
        max_score = 5

        # 1. Partners publicly listed (20 points)
        if row['Partners Public'] == 'Yes':
            score += 1

        # 2. Fee information public (20 points)
        if str(row['Fees Public']).lower() in ['yes', 'partial', 'variable']:
            score += 1

        # 3. Specific fee amounts posted (20 points)
        fee_mean = fee_df_all[fee_df_all['Institution'] ==
                              row['Institution']]['Fee Mean'].values
        if len(fee_mean) > 0 and pd.notna(fee_mean[0]):
            score += 1

        # 4. Tier information available (20 points)
        if row['Tiers'] == 'Yes' and pd.notna(row['Tier Names']):
            score += 1

        # 5. Benefits clearly listed (20 points)
        if pd.notna(row['Benefits']) and len(str(row['Benefits'])) > 10:
            score += 1

        return (score / max_score) * 100

    df['Transparency_Score'] = df.apply(calculate_transparency_score, axis=1)

    # Sort by score
    transparency_sorted = df.sort_values('Transparency_Score',
                                         ascending=True)

    fig, ax = plt.subplots(figsize=(14, 12))

    # Color gradient based on score
    colors = plt.cm.RdYlGn(
        transparency_sorted['Transparency_Score'] / 100)

    bars = ax.barh(transparency_sorted['Institution'],
                   transparency_sorted['Transparency_Score'],
                   color=colors, edgecolor='black', linewidth=0.5)

    # Add score labels
    for i, (inst, score) in enumerate(
            zip(transparency_sorted['Institution'],
                transparency_sorted['Transparency_Score'])):
        ax.text(score + 1, i, f'{score:.0f}%', va='center', fontsize=9)

    # Add reference lines
    ax.axvline(x=60, color='orange', linestyle='--',
               alpha=0.5, label='60% threshold')
    ax.axvline(x=80, color='green', linestyle='--',
               alpha=0.5, label='80% threshold')

    ax.set_xlabel('Transparency Score (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Institution', fontsize=12, fontweight='bold')
    title_text = ('Program Transparency Score\n'
                  '(Partners + Fee Info + Amounts + Tiers + Benefits)')
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 105)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/transparency_score.png', bbox_inches='tight')
    plt.close()

    # --- Generate Summary Statistics ---
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTotal Institutions Analyzed: {len(df)}")
    print(f"Institutions with Tiered Programs: {(df['Tiers'] == 'Yes').sum()}")
    print(f"Institutions indicating fees publicly: {fee_public_mask.sum()}")
    print(f"\nAverage Number of Public Partners: {df['No Of Public Partners'].mean():.1f}")
    print(f"Median Number of Public Partners: {df['No Of Public Partners'].median():.0f}")
    print(f"Max Number of Public Partners: {df['No Of Public Partners'].max():.0f} ({df.loc[df['No Of Public Partners'].idxmax(), 'Institution']})")
    
    if not fee_with_amounts.empty:
        print(f"\nAverage Membership Fee (where amount posted): ${fee_with_amounts['Fee Mean'].mean():,.0f}")
        print(f"Median Membership Fee (where amount posted): ${fee_with_amounts['Fee Mean'].median():,.0f}")
        print(f"Fee Range (where posted): ${fee_with_amounts['Fee Mean'].min():,.0f} - ${fee_with_amounts['Fee Mean'].max():,.0f}")
    else:
        print("\nNo numeric fee amounts were available for averaging.")
    
    print(f"\nMost Common Technology Sector: {sector_counts.index[0]} ({sector_counts.iloc[0]} institutions)")
    print(f"Most Common Benefit: {benefit_counts.index[0]} ({benefit_counts.iloc[0]} institutions)")
    print(f"Most Frequent Corporate Partner: {partner_counts.index[0]} ({partner_counts.iloc[0]} partnerships)")

    print(f"\nAverage Transparency Score: {df['Transparency_Score'].mean():.1f}%")
    highest_trans_inst = df.loc[df['Transparency_Score'].idxmax(), 'Institution']
    highest_trans_score = df['Transparency_Score'].max()
    print(f"Highest Transparency: {highest_trans_inst} ({highest_trans_score:.0f}%)")
    print(f"Institutions with 80%+ Transparency: {(df['Transparency_Score'] >= 80).sum()}")

    # --- Save processed data for downstream inspection ---
    proc = df.copy()
    proc['fees_public_flag'] = fee_public_mask
    proc = proc.merge(fee_df_all[['Institution', 'Fee Mean']], on='Institution', how='left')
    # attach cleaned sectors as a joined string
    proc['cleaned_sectors'] = sectors_series.apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    # Include transparency score
    proc['Transparency_Score'] = df['Transparency_Score']
    proc.to_csv('processed_data.csv', index=False)

    print("\n" + "="*60)
    print("Analysis complete. All plots saved in the 'plots' directory.")
    print("Processed data saved to 'processed_data.csv'.")
    print("="*60)


if __name__ == '__main__':
    analyze_data()
