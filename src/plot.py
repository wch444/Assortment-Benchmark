import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_instance_summary(
    instances,
    group_cols,
    color_map=plt.cm.Set3,
    label_fontsize=9
):
    """
    
    Generic visualization function for instance distributions.
    
    It automatically generates bar plots to summarize how instances are distributed
    across each individual dimension and their combined parameter settings.

    Parameters
    ----------
    instances : list[dict] or list[object]
    group_cols : list[str], the columns used for grouping and summarization.
      
    color_map : matplotlib colormap, default=plt.cm.Set3
        The colormap used for visualizing categorical distributions.
    """

    # ==========================================================
    # Extract key attributes from instance list (supports dict or object)
    # ==========================================================
    attr_names = {col: [] for col in group_cols + ['seed']}
    for inst in instances:
        for col in attr_names:
            attr_names[col].append(getattr(inst, col, None))

    df_summary = pd.DataFrame(attr_names)

    print("=" * 80)
    print("DATASET STRUCTURE SUMMARY")
    print("=" * 80)

    # ==========================================================
    # Plot marginal distributions for each group column
    # ==========================================================
    n_cols = len(group_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(18, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(
        'Instance Distribution Across Different Dimensions',
        fontsize=14, fontweight='bold', y=1.02
    )

    for i, col in enumerate(group_cols):
        counts = df_summary[col].value_counts().sort_index()
        x = np.arange(len(counts))
        bars = axes[i].bar(
            x,
            counts.values,
            color=color_map(np.arange(len(counts)) % 12),
            edgecolor='black',
            alpha=0.75
        )
        axes[i].set_title(f'Group By {col}', fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Number of Instances')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(counts.index.astype(str), rotation=0)
        axes[i].grid(True, alpha=0.3, axis='y')

        # Add text labels on top of each bar
        for bar, val in zip(bars, counts.values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(val),
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=9
            )

    plt.tight_layout()
    plt.show()

    # ==========================================================
    # Compute grouped statistics and visualize combinations
    # ==========================================================
    grouped = (
        df_summary.groupby(group_cols)
        .size()
        .reset_index(name='num_instances')
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    # Create combined textual label such as "(m, n, cap_rate)"
    grouped['label'] = grouped.apply(
        lambda r: '(' + ', '.join(str(r[c]) for c in group_cols) + ')', axis=1
    )

    fig, ax = plt.subplots(figsize=(max(8, len(grouped) * 0.6), 4))
    x_pos = np.arange(len(grouped))
    bars = ax.bar(
        x_pos,
        grouped['num_instances'],
        color=color_map(np.arange(len(grouped)) % 12),
        edgecolor='black',
        alpha=0.75
    )

    ax.set_xlabel(
        f"Combinations of ({', '.join(group_cols)})",
        fontsize=12, fontweight='bold'
    )
    ax.set_ylabel('Number of Instances', fontsize=12, fontweight='bold')
    ax.set_title(
        'Instance Distribution by Parameter Combinations',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped['label'], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels above each bar
    for bar, val in zip(bars, grouped['num_instances']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(val),
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=9
        )

    plt.tight_layout()
    plt.show()


    



def explore_instance(data, model_type):
    """
    Visualize and summarize a single instance of an assortment optimization dataset.

    This function provides a detailed overview of the instance attributes.

    Parameters
    ----------
    data : object
        A single instance object containing model parameters.
    model_type : str
        The model type indicator. Accepts "NL" or "MMNL".
        Determines which fields are used and how the data are visualized.
    """
    # =============================================================================
    # BASIC INFO
    # =============================================================================
    print("=" * 80)
    print("INSTANCE OVERVIEW")
    print("=" * 80)

    # 通用信息
    m = getattr(data, "m", None)
    n = getattr(data, "n", None)
    seed = getattr(data, "seed", None)
    cap_rate = getattr(data, "cap_rate", None)
    max_rev = getattr(data, "max_rev", None)

    if model_type.upper() == "NL":
        print(f"Number of customer segments (m): {m}")
        print(f"Number of products (n): {n}")
    else:
        print(f"Number of nests (m): {m}")
        print(f"Number of products per nest (n): {n}")
        print(f"Total number of products: {m * n if m and n else 'N/A'}")

    print(f"Random seed: {seed}")
    print(f"Capacity rate: {cap_rate}")
    if max_rev is not None:
        print(f"Maximum revenue (optimal): {max_rev:.4f}")

    # =============================================================================
    # DATA COMPONENTS - DIMENSIONS
    # =============================================================================
    print("\n" + "=" * 80)
    print("DATA COMPONENTS - DIMENSIONS")
    print("=" * 80)

    if model_type.upper() == "MMNL":
        u = data.u
        price = data.price.flatten()
        v0 = np.array(data.v0).flatten()
        omega = np.array(data.omega).flatten()

        print(f"1. Utility Matrix (u): shape {u.shape}, range: [{u.flatten().min():.4f}, {u.flatten().max():.4f}]")
        print(f"2. Product Prices (price): shape {data.price.shape},range: [{price.min():.4f}, {price.max():.4f}]")
        print(f"3. No-Purchase Utility value (v0): {data.v0}")
        print(f"4. Customer Segment Probabilities (omega): shape {data.omega.shape}, sum: {omega.sum():.4f} (should be 1.0)")

    else:
        u = data.v
        price = data.price.flatten()
        gamma = np.array(data.gamma).flatten()
        vi0 = data.vi0
        vi0_flat = vi0.flatten() if isinstance(vi0, np.ndarray) else np.array([vi0])
        v0 = data.v0

        print(f"1. Utility Matrix (v): shape {u.shape}, range: [{u.flatten().min():.4f}, {u.flatten().max():.4f}]")
        print(f"2. Product Prices (price): shape {data.price.shape},range: [{price.min():.4f}, {price.max():.4f}]")
        print(f"3. Dissimilarity Parameter (gamma): shape {data.gamma.shape},range: [{gamma.min():.4f}, {gamma.max():.4f}]")
        print(f"4. No-Purchase Utility (v0): {v0:.4f}")
        if isinstance(vi0, np.ndarray) and vi0.size > 1:
            print(f"5. Within-Nest No-Purchase Utility (vi0): shape: {vi0.shape}, range: [{vi0_flat.min():.4f}, {vi0_flat.max():.4f}]")
        else:
            print(f"5. Within-Nest No-Purchase Utility (vi0): {vi0:.4f} (uniform across nests)")

    print("\n" + "=" * 80)
    print("DATA VISUALIZATION")
    print("=" * 80)


    fig = plt.figure(figsize=(16, 10))
    # (1) Utility Matrix Heatmap
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(u, cmap="YlOrRd", cbar_kws={'label': 'Utility'}, ax=ax1)
    ax1.set_title("Utility Matrix", fontweight="bold")
    ax1.set_xlabel("Products"  if model_type.upper() == "MMNL" else "Products per Nest")
    ax1.set_ylabel("Customer Segments" if model_type.upper() == "MMNL" else "Nests")

    # (2) Price Distribution

    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(price, bins=20, edgecolor="black", color="skyblue", alpha=0.7)
    ax2.set_title("Price Distribution", fontweight="bold")
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    # (3) Parameter Distributions (v0 / gamma)
    if  model_type.upper() == "MMNL":
        ax3 = plt.subplot(2, 3, 3)
        ax3.bar(range(len(v0)), v0, color="coral", edgecolor="black", alpha=0.7)
        ax3.set_title('No-Purchase Utility by Segment', fontweight='bold')
        ax3.set_xlabel('Customer Segment')
        ax3.set_ylabel('v0 (No-Purchase Utility)')
        ax3.set_xticks(range(len(v0)))
        ax3.set_xticklabels([f'S{i+1}' for i in range(len(v0))])
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.legend()
        # (4) Segment Probability Pie Chart
        ax4 = plt.subplot(2, 3, 4)
        colors = plt.cm.Set3(range(len(omega)))
        ax4.pie(omega, labels=[f'Segment {i+1}' for i in range(len(omega))], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax4.set_title('Customer Segment Distribution', fontweight='bold')
    elif model_type.upper() == "NL":
        ax3 = plt.subplot(2, 3, 3)
        ax3.bar(range(len(gamma)), gamma, color="coral", edgecolor="black", alpha=0.7)
        ax3.set_title('Dissimilarity Parameter by Nest', fontweight='bold')
        ax3.set_xlabel('Nest')
        ax3.set_ylabel('gamma (Dissimilarity)')
        ax3.set_xticks(range(data.m))
        ax3.set_xticklabels([f'N{i+1}' for i in range(data.m)])
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.legend()

        # (4) Within-Nest No-Purchase Utility Visualization
        ax4 = plt.subplot(2, 3, 4)
        if isinstance(data.vi0, np.ndarray) and data.vi0.size > 1:
            colors = plt.cm.Set3(range(data.m))
            ax4.pie(vi0, labels=[f'Nest {i+1}' for i in range(data.m)], 
                    autopct=lambda pct: f'{pct:.1f}%\n({vi0[int(pct/100.*len(vi0))]:.2f})', 
                    colors=colors, startangle=90)
            ax4.set_title('Within-Nest No-Purchase Utility (vi0)', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, f'vi0 = {data.vi0:.4f}\n(Full Capture)', 
                    ha='center', va='center', fontsize=14, transform=ax4.transAxes)
            ax4.set_title('Within-Nest No-Purchase Utility', fontweight='bold')
            ax4.axis('off')
    

    # (5) Utility Statistics
    ax5 = plt.subplot(2, 3, 5)
 
    stats = pd.DataFrame({
        "Mean": u.mean(axis=1),
        "Std": u.std(axis=1),
        "Min": u.min(axis=1),
        "Max": u.max(axis=1)
    })
    x = np.arange(u.shape[0])
    w = 0.2
    for i, col in enumerate(stats.columns):
        ax5.bar(x + (i - 1.5) * w, stats[col], w, label=col, alpha=0.8)
    ax5.set_title("Utility Statistics by Segment" if model_type.upper() == "MMNL" else 'Utility Statistics by Nest', fontweight="bold")
    ax5.set_xlabel('Customer Segment' if model_type.upper() == "MMNL" else 'Nest')
    ax5.set_ylabel('Utility Value')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'{i+1}' for i in range(data.m)])
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")
  

    # (6) Price vs Average Utility
    ax6 = plt.subplot(2, 3, 6)
    avg_utility = u.mean(axis=0)
    if model_type.upper() == "NL":
        avg_price = data.price.mean(axis=0)
    else:
        avg_price = price
    ax6.scatter(avg_price, avg_utility, alpha=0.6, s=50, color="green")
    ax6.set_title("Price vs Average Utility", fontweight="bold")
    ax6.set_xlabel("Price" if model_type.upper() == "MMNL" else "Average Price Across Nests")
    ax6.set_ylabel("Average Utility Across Segments" if model_type.upper() == "MMNL" else "Average Utility Across Nests")
    ax6.grid(True, alpha=0.3)
   
    plt.tight_layout()
    plt.show()






def export_summary_statistics(
    df_results,
    output_dir,
    file_name,
    method_col,
    group_cols=["m", "n"],
    aggfuncs=("mean", "std"),
):
    """
    Export summarized experiment results to an Excel file.

    Parameters
    ----------
    df_results : pd.DataFrame.  A DataFrame containing all experimental results. Must include the method column (e.g., "rev_method") and metric column (e.g., "gap_method").
    output_dir : str. The directory path where the Excel file will be saved. Will be created automatically if it does not exist.
    file_name : str. Name of the exported Excel file .
    method_col : str. CColumn name representing the different experimental methods (e.g., "rev_method", "vi0_method").
    group_cols : list[str]. Columns used for grouping and aggregation, e.g., ["m", "n"]. Currently supports up to two grouping columns.
    aggfuncs : tuple, optional. Aggregation functions to compute (e.g., ("mean", "std")).

    Raises
    ------
    ValueError
        If more than two grouping columns are provided.

    Output
    ------
    An Excel file with the following sheets:
        - "All Results": The full dataset.
        - "{method}_{func}": Pivot tables for each method and aggregation function.
        - "{method}_Details": Descriptive statistics for each (m, n) combination.
        - "Comparison": Cross-method mean performance comparison.

        """
    if len(group_cols) > 2:
        raise ValueError("Currently only supports up to 2 grouping columns.")
    
    df_results = df_results.copy()
    detail_group_col = "_".join(group_cols) + "_pair"
    df_results[detail_group_col] = df_results[group_cols].apply(
        lambda row: "(" + ", ".join(str(v) for v in row.values) + ")", axis=1
    )

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)

    # Write results to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Raw data
        df_results.to_excel(writer, sheet_name="All Results", index=False)

        # Generate method-specific statistics
        for method_value in df_results[method_col].unique():
            df_subset = df_results[df_results[method_col] == method_value]

            # Compute aggregation functions (e.g., mean, std)
            for func in aggfuncs:
                pivot = df_subset.pivot_table(
                    values="gap_method",
                    index=group_cols[0],
                    columns=group_cols[1] if len(group_cols) > 1 else None,
                    aggfunc=func
                )
                sheet_name = f"{method_value}_{func.capitalize()}"
                pivot.to_excel(writer, sheet_name=sheet_name)

            # Add detailed descriptive statistics
            summary_detail = df_subset.groupby(detail_group_col)["gap_method"].describe()
            summary_detail.to_excel(writer, sheet_name=f"{method_value}_Details")

        # Compare mean performance across different methods
        pivot_compare = df_results.pivot_table(
            values="gap_method",
            index=group_cols,
            columns=method_col,
            aggfunc="mean"
        )
        pivot_compare.to_excel(writer, sheet_name="Comparison")
    



def plot_comparison_boxplots(df,method_col,group_col):
    """
    Generic boxplot comparison function for visualizing performance differences 
    across methods and grouped settings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experimental results.
    method_col : str
        Column name indicating different methods (e.g., 'vi0_method' or 'rev_method').
    group_col : str
        Column name used for x-axis grouping (e.g., '(m,n)' or combined group key).
    """
    methods = df[method_col].unique()
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(20, 6), squeeze=False)
    axes = axes.flatten()

    for idx, method in enumerate(methods):
        df_subset = df[df[method_col] == method]

        sns.boxplot(
            data=df_subset,
            x=group_col,
            y="gap_method",
            ax=axes[idx],
            palette="Set3",
            width=0.6
        )

        axes[idx].set_xlabel(f"{group_col} Combinations", fontsize=11, fontweight="bold")
        axes[idx].set_ylabel("Optimality Gap (%)", fontsize=11, fontweight="bold")
        axes[idx].set_title(f"{method}", fontsize=12, fontweight="bold")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].axhline(
            y=0, color="r", linestyle="--", alpha=0.5, linewidth=1, label="Optimal (0% gap)"
        )
        axes[idx].legend()

    method_label = method_col.replace("_", " ").title()
    plt.suptitle(f"Algorithm Performance Comparison Across {method_label}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

    for method in methods:
        df_method = df[df[method_col] == method]
        # Print in requested format
        print(f"\n{method} Performance Summary:")  # [MODIFIED]
        print(df_method.groupby(group_col)["gap_method"].agg(["mean", "std", "min", "max"]).round(4)) 




