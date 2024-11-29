import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import json
import os
import warnings
from datetime import datetime
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Suppress warnings
warnings.filterwarnings('ignore')

class CognitiveBehavioralTherapyAnalysis:
    """
    A class to analyze Cognitive Behavioral Therapy session evaluations using mixed effects models.
    """
    
    def __init__(self, data_path: str = "evaluation_results/all_scores_with_patients.csv"):
        """
        Initialize the analysis with the path to evaluation results.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing evaluation results
        """
        self.dataframe = pd.read_csv(data_path)
        self.categories = [
            "Agenda",
            "Feedback",
            "Understanding",
            "Interpersonal Effectiveness",
            "Collaboration",
            "Pacing and Efficient Use of Time",
            "Guided Discovery",
            "Focusing on Key Cognitions or Behaviors",
            "Strategy for Change",
            "Application of Cognitive-Behavioral Techniques",
            "Homework"
        ]
        self.results_directory = "analysis_results"
        os.makedirs(self.results_directory, exist_ok=True)
        
        # Create timestamp for unique file naming
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare data
        self._prepare_data()

    def _prepare_data(self) -> None:
        """
        Prepare the data for analysis by extracting relevant information and creating necessary columns.
        """
        # Ensure all required columns exist
        required_columns = ['group', 'id', 'patient_id'] + self.categories
        missing_columns = [col for col in required_columns if col not in self.dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Report missing data before handling
        missing_counts = self.dataframe[self.categories].isnull().sum()
        print("\nMissing data counts per category before handling:")
        print(missing_counts)

        # Remove rows with missing category scores
        self.dataframe.dropna(subset=self.categories + ['patient_id'], inplace=True)

        # Print unique groups for debugging
        print("\nProcessing groups:", self.dataframe['group'].unique())

        # Extract model type and variant from the more complex group format
        try:
            # Create model and variant columns based on the group name pattern
            def parse_group(group):
                parts = group.split('_')
                model = parts[0]  # Llama, Mistral, Qwen
                variant = 'instruct' if 'Instruct' in group else 'finetuned'
                return pd.Series([model, variant])
            
            self.dataframe[['model', 'variant']] = self.dataframe['group'].apply(parse_group)
            
            print("\nExtracted models:", self.dataframe['model'].unique())
            print("Extracted variants:", self.dataframe['variant'].unique())
            
        except Exception as e:
            print(f"Error processing group format: {str(e)}")
            print("Group values found:", self.dataframe['group'].unique())
            raise ValueError(f"Error processing group format: {str(e)}")
        
        # Extract session number from ID
        try:
            session_numbers = self.dataframe['id'].str.extract(r'^(\d+)_')
            if session_numbers.isnull().all().all():
                print("Warning: Could not extract session numbers from ID. Using sequential numbering.")
                self.dataframe['session'] = range(len(self.dataframe))
            else:
                self.dataframe['session'] = session_numbers.astype(float).fillna(-1).astype(int)
        except Exception as e:
            print(f"Error extracting session numbers: {str(e)}")
            print("ID format found:", self.dataframe['id'].head())
            self.dataframe['session'] = range(len(self.dataframe))
        
        # Calculate total CTRS score
        self.dataframe['total_score'] = self.dataframe[self.categories].sum(axis=1)
        
        # Ensure patient_id is properly formatted
        if not isinstance(self.dataframe['patient_id'].dtype, pd.CategoricalDtype):
            self.dataframe['patient_id'] = self.dataframe['patient_id'].astype('category')
        
        # Ensure categorical variables are of type 'category'
        self.dataframe['model'] = self.dataframe['model'].astype('category')
        self.dataframe['variant'] = self.dataframe['variant'].astype('category')
        self.dataframe['session'] = self.dataframe['session'].astype(int)

        # Remove any rows with missing values in key columns
        self.dataframe = self.dataframe.dropna(subset=['total_score', 'model', 'variant', 'session', 'patient_id'])

        # Sort by session number and patient_id
        self.dataframe = self.dataframe.sort_values(['patient_id', 'session'])

        # Validate that we have enough data for mixed effects modeling
        if len(self.dataframe['patient_id'].unique()) < 2:
            raise ValueError("Need at least 2 patients for mixed effects modeling")
        if len(self.dataframe) < 4:
            raise ValueError("Need at least 4 observations for mixed effects modeling")
        
        # Print summary of prepared data
        print("\nData Preparation Summary:")
        print(f"Total observations: {len(self.dataframe)}")
        print(f"Number of unique patients: {len(self.dataframe['patient_id'].unique())}")
        print(f"Number of unique models: {len(self.dataframe['model'].unique())}")
        print(f"Models: {', '.join(sorted(self.dataframe['model'].unique()))}")
        print(f"Number of unique variants: {len(self.dataframe['variant'].unique())}")
        print(f"Variants: {', '.join(sorted(self.dataframe['variant'].unique()))}")
        print(f"Session range: {self.dataframe['session'].min()} to {self.dataframe['session'].max()}")

        # Check normality of total_score
        plt.figure(figsize=(8,6))
        stats.probplot(self.dataframe['total_score'], dist="norm", plot=plt)
        plt.title('Normal Q-Q Plot of Total CTRS Scores')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_directory, f'qqplot_total_score_{self.timestamp}.png'))
        plt.close()

    def _validate_model_structure(self, model_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the data structure for mixed effects modeling.
        
        Parameters:
        -----------
        model_data : pd.DataFrame
            The prepared data for modeling
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing validation results
        """
        validation = {
            'is_valid': True,
            'messages': [],
            'warnings': []
        }
        
        # Check sample size requirements
        n_total = len(model_data)
        n_groups = len(model_data['patient_id'].unique())
        
        if n_total < 50:
            validation['warnings'].append(f"Small sample size ({n_total} < 50 observations)")
        
        if n_groups < 5:
            validation['warnings'].append(f"Small number of groups ({n_groups} < 5 groups)")
        
        # Check for balance
        group_sizes = model_data.groupby('patient_id').size()
        if group_sizes.std() / group_sizes.mean() > 0.5:
            validation['warnings'].append("Highly unbalanced design (high variation in group sizes)")
        
        # Check for collinearity
        model_dummies = pd.get_dummies(model_data[['model', 'variant']])
        corr_matrix = model_dummies.corr()
        high_corr = np.where(np.abs(corr_matrix) > 0.8)
        high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i,j])
                        for i, j in zip(*high_corr) if i < j]
        
        if high_corr_pairs:
            validation['warnings'].append("High collinearity detected between predictors")
            for var1, var2, corr in high_corr_pairs:
                validation['warnings'].append(f"  - {var1} and {var2}: correlation = {corr:.3f}")
        
        # Check for complete separation
        for col in ['model', 'variant']:
            group_cross = pd.crosstab(model_data['patient_id'], model_data[col])
            if (group_cross == 0).any().any():
                validation['warnings'].append(f"Complete separation detected in {col}")
        
        return validation

    def run_mixed_effects_model(self) -> Dict[str, Any]:
        """
        Fit a mixed effects model and report the results.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing the mixed effects model results
        """
        try:
            # Prepare data for mixed effects model
            model_data = self.dataframe.copy()
            
            # Ensure numeric types
            model_data['session'] = pd.to_numeric(model_data['session'], errors='coerce')
            model_data['total_score'] = pd.to_numeric(model_data['total_score'], errors='coerce')
            
            # Remove any rows with NaN values
            model_data = model_data.dropna(subset=['session', 'total_score', 'model', 'variant', 'patient_id'])
            
            # Center the session variable
            model_data['session_centered'] = (model_data['session'] - model_data['session'].mean()) / model_data['session'].std()
            
            # Specify the formula with interaction terms
            formula = "total_score ~ C(model) + C(variant) + session_centered"
            
            # Fit the mixed effects model with random slopes
            md = smf.mixedlm(
                formula,
                data=model_data,
                groups="patient_id",
                re_formula="1"
            )
            
            # Try to fit the model, handle convergence issues
            try:
                mdf = md.fit(reml=False, method='lbfgs')
            except Exception as e:
                print(f"Initial model fit failed: {e}")
                # Try fitting with different methods or simplifying the model
                try:
                    mdf = md.fit(reml=False, method='powell')
                except Exception as e:
                    print(f"Model fit with 'powell' method failed: {e}")
                    # Simplify the model
                    formula = "total_score ~ session_centered + C(model) + C(variant)"
                    md = smf.mixedlm(
                        formula,
                        data=model_data,
                        groups="patient_id",
                        re_formula="~session_centered"
                    )
                    mdf = md.fit()
            
            # Store the results
            results = {
                'model_summary': str(mdf.summary()),
                'fixed_effects': mdf.fe_params.to_dict(),
                'random_effects': {str(k): v.values.tolist() for k, v in mdf.random_effects.items()},
                'pvalues': mdf.pvalues.to_dict(),
                'aic': mdf.aic,
                'bic': mdf.bic,
                'n_observations': len(model_data),
                'n_groups': len(model_data['patient_id'].unique()),
                'formula_used': formula
            }
            
            # Adjust p-values for multiple testing using Bonferroni correction
            pvals = mdf.pvalues
            adjusted_pvals = multipletests(pvals, method='bonferroni')[1]
            results['adjusted_pvalues'] = dict(zip(pvals.index, adjusted_pvals))
            
            # Add confidence intervals for fixed effects
            conf_int = mdf.conf_int()
            conf_int.columns = ['lower', 'upper']
            results['confidence_intervals'] = conf_int.to_dict(orient='index')
            
            # Add model diagnostics
            results['diagnostics'] = {
                'number_of_models': len(model_data['model'].unique()),
                'number_of_variants': len(model_data['variant'].unique()),
                'sessions_per_patient': model_data.groupby('patient_id').size().to_dict(),
                'variance_explained': mdf.fittedvalues.var() / model_data['total_score'].var()
            }
            
            # Residual diagnostics
            residuals = mdf.resid
            fitted = mdf.fittedvalues
            
            # Plot residuals vs fitted values
            plt.figure(figsize=(8,6))
            sns.residplot(x=fitted, y=residuals, lowess=True)
            plt.xlabel('Fitted values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Fitted')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_directory, f'residuals_vs_fitted_{self.timestamp}.png'))
            plt.close()
            
            # Q-Q plot
            plt.figure(figsize=(8,6))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('Normal Q-Q')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_directory, f'normal_qq_{self.timestamp}.png'))
            plt.close()
            
            # Save residuals and fitted values
            results['residuals'] = residuals.tolist()
            results['fitted_values'] = fitted.tolist()
            
        except Exception as e:
            results = {
                'error': str(e),
                'error_type': str(type(e).__name__),
                'data_shape': self.dataframe.shape,
                'groups_count': len(self.dataframe['patient_id'].unique()),
                'observations_per_group': self.dataframe.groupby('patient_id').size().to_dict(),
                'model_description': {
                    'unique_models': sorted(model_data['model'].unique()),
                    'unique_variants': sorted(model_data['variant'].unique()),
                    'session_range': f"{model_data['session'].min()} to {model_data['session'].max()}"
                }
            }
        
        return results

    def create_visualizations(self) -> None:
        """Create comprehensive set of visualizations for the analysis."""
        try:
            # Set up the visualization style with larger font sizes
            plt.style.use('default')
            sns.set_palette("husl")
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12
            })

            # Rename the groups
            group_mapping = {
                'Mistral_Finetune_Simulation_Nov_6': 'Mistral_CBT',
                'Mistral_Instruct_Simulation_Nov_2': 'Mistral_Instruct',
                'Llama_Finetune_Simulation_Nov_6': 'Llama_CBT',
                'Llama_Instruct_Simulation_Nov_2': 'Llama_Instruct',
                'Qwen_Finetune_Simulation_Nov_16': 'Qwen_CBT',
                'Qwen_Instruct_Simulation_Nov_15': 'Qwen_Instruct'
            }
            
            viz_data = self.dataframe.copy()
            viz_data['group'] = viz_data['group'].replace(group_mapping)
            
            # 1. Box plots of total CTRS scores by model variant
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=viz_data, x='group', y='total_score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_directory, f'total_scores_boxplot_{self.timestamp}.png'))
            plt.close()

            # 2. Combined radar plot for category-specific scores across all groups
            try:
                plt.figure(figsize=(12, 12))
                ax = plt.subplot(111, polar=True)
                labels = self.categories
                num_labels = len(labels)
                angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False)
                angles = np.concatenate((angles, [angles[0]]))
                
                for group in viz_data['group'].unique():
                    values = viz_data[viz_data['group'] == group][self.categories].mean()
                    values = np.concatenate((values, [values[0]]))
                    ax.plot(angles, values, label=group, linewidth=2)
                    ax.fill(angles, values, alpha=0.1)
                
                ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_directory, f'combined_radar_plot_{self.timestamp}.png'))
                plt.close()
            except Exception as e:
                print(f"Error creating combined radar plot: {str(e)}")
                raise

            # 3. Line plots for session progression with confidence intervals
            plt.figure(figsize=(12, 6))
            for group in viz_data['group'].unique():
                group_data = viz_data[viz_data['group'] == group]
                
                session_stats = group_data.groupby('session')['total_score'].agg(['mean', 'std', 'count']).reset_index()
                session_stats['sem'] = session_stats['std'] / np.sqrt(session_stats['count'])
                session_stats['ci'] = session_stats['sem'] * stats.t.ppf(1 - 0.025, session_stats['count'] - 1)
                
                plt.plot(session_stats['session'], session_stats['mean'], 
                        label=group, marker='o', linewidth=2)
                plt.fill_between(
                    session_stats['session'],
                    session_stats['mean'] - session_stats['ci'],
                    session_stats['mean'] + session_stats['ci'],
                    alpha=0.2
                )
            
            plt.xlabel('Session Number')
            plt.ylabel('Average Total CTRS Score')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_directory, f'session_progression_{self.timestamp}.png'))
            plt.close()

            # 4. Heatmap of category scores
            plt.figure(figsize=(15, 10))
            category_scores = viz_data.groupby('group')[self.categories].mean()
            
            # Create heatmap with adjusted y-axis labels
            g = sns.heatmap(category_scores, annot=True, cmap='YlOrRd', fmt='.2f', annot_kws={'size': 12})
            
            # Rotate y-axis labels to horizontal and adjust their position
            g.set_yticklabels(g.get_yticklabels(), rotation=0)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_directory, f'category_scores_heatmap_{self.timestamp}.png'), 
                        bbox_inches='tight',  # This ensures no labels are cut off
                        dpi=300)  # Higher DPI for better quality
            plt.close()

            # 5. Distribution plots for each category (only create plots for existing categories)
            num_categories = len(self.categories)
            num_rows = (num_categories + 2) // 3  # Calculate needed rows
            fig, axes = plt.subplots(num_rows, 3, figsize=(20, 6 * num_rows))
            axes = axes.ravel()
            
            for index, category in enumerate(self.categories):
                sns.violinplot(data=viz_data, x='group', y=category, ax=axes[index])
                axes[index].set_xticklabels(axes[index].get_xticklabels(), rotation=45)
                axes[index].set_title(f'Distribution of {category} Scores')
            
            # Remove any unused subplots
            for idx in range(len(self.categories), len(axes)):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_directory, f'category_distributions_{self.timestamp}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            raise

    def generate_summary_report(self, mixed_effects_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive summary report of all analyses.
        
        Parameters:
        -----------
        mixed_effects_results : Dict[str, Any]
            The results from the mixed effects model
        
        Returns:
        --------
        str
            Formatted summary report
        """
        summary = []
        summary.append("=" * 80)
        summary.append("COGNITIVE BEHAVIORAL THERAPY ANALYSIS SUMMARY REPORT")
        summary.append("=" * 80 + "\n")

        # Data Overview
        summary.append("DATA OVERVIEW")
        summary.append("-" * 40)
        summary.append(f"Total number of sessions: {len(self.dataframe)}")
        summary.append(f"Number of unique patients: {len(self.dataframe['patient_id'].unique())}")
        summary.append(f"Number of groups: {len(self.dataframe['group'].unique())}")
        summary.append("\nGroups:")
        for group in self.dataframe['group'].unique():
            n_sessions = len(self.dataframe[self.dataframe['group'] == group])
            n_patients = len(self.dataframe[self.dataframe['group'] == group]['patient_id'].unique())
            summary.append(f"  {group}: {n_sessions} sessions, {n_patients} patients")

        # Sessions per patient statistics
        sessions_per_patient = self.dataframe.groupby('patient_id').size()
        summary.append(f"\nSessions per patient:")
        summary.append(f"  Minimum: {sessions_per_patient.min()}")
        summary.append(f"  Maximum: {sessions_per_patient.max()}")
        summary.append(f"  Average: {sessions_per_patient.mean():.1f}")
        summary.append(f"  Median: {sessions_per_patient.median()}")

        # Basic Statistics
        summary.append("\nBASIC STATISTICS")
        summary.append("-" * 40)
        for group in self.dataframe['group'].unique():
            group_data = self.dataframe[self.dataframe['group'] == group]
            summary.append(f"\nGroup: {group}")
            summary.append(f"Number of sessions: {len(group_data)}")
            summary.append(f"Average total score: {group_data['total_score'].mean():.2f}")
            summary.append(f"Standard deviation: {group_data['total_score'].std():.2f}")
            summary.append(f"Min score: {group_data['total_score'].min():.2f}")
            summary.append(f"Max score: {group_data['total_score'].max():.2f}")
            
            # Category-specific statistics
            summary.append("\nCategory Scores:")
            for category in self.categories:
                mean_score = group_data[category].mean()
                std_score = group_data[category].std()
                summary.append(f"  {category}: {mean_score:.2f} ± {std_score:.2f}")

        # Mixed Effects Model Results
        summary.append("\nMIXED EFFECTS MODEL RESULTS")
        summary.append("-" * 40)
        if 'error' in mixed_effects_results:
            summary.append(f"Error fitting mixed effects model: {mixed_effects_results['error']}")
            summary.append(f"Error type: {mixed_effects_results.get('error_type', 'Unknown')}")
            summary.append(f"Data shape: {mixed_effects_results.get('data_shape', 'Unknown')}")
            summary.append(f"Number of groups: {mixed_effects_results.get('groups_count', 'Unknown')}")
            summary.append("\nObservations per group:")
            for group, count in mixed_effects_results.get('observations_per_group', {}).items():
                summary.append(f"  {group}: {count}")
        else:
            summary.append(mixed_effects_results['model_summary'])
            # Include effect sizes (estimates and confidence intervals)
            summary.append("\nFixed Effects Estimates and Confidence Intervals:")
            fe_params = mixed_effects_results['fixed_effects']
            conf_ints = mixed_effects_results['confidence_intervals']
            for param in fe_params:
                estimate = fe_params[param]
                ci_lower = conf_ints[param]['lower']
                ci_upper = conf_ints[param]['upper']
                summary.append(f"  {param}: Estimate = {estimate:.4f}, 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]")

        return "\n".join(summary)

    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete analysis suite including mixed effects model and visualizations.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing all analysis results
        """
        try:
            # Run mixed effects model
            mixed_effects_results = self.run_mixed_effects_model()

            # Create results dictionary
            results = {
                'mixed_effects_model': mixed_effects_results,
                'data_summary': {
                    'n_observations': len(self.dataframe),
                    'n_patients': len(self.dataframe['patient_id'].unique()),
                    'n_groups': len(self.dataframe['group'].unique()),
                    'sessions_per_patient': self.dataframe.groupby('patient_id').size().to_dict(),
                    'scores_by_group': {
                        group: {
                            'mean': float(group_data['total_score'].mean()),
                            'std': float(group_data['total_score'].std()),
                            'min': float(group_data['total_score'].min()),
                            'max': float(group_data['total_score'].max())
                        }
                        for group, group_data in self.dataframe.groupby('group')
                    }
                }
            }
            
            # Generate and save summary report
            summary_report = self.generate_summary_report(mixed_effects_results)
            report_path = os.path.join(self.results_directory, f'summary_report_{self.timestamp}.txt')
            with open(report_path, 'w') as file:
                file.write(summary_report)
            
            # Save numerical results
            results_path = os.path.join(self.results_directory, f'analysis_results_{self.timestamp}.json')
            with open(results_path, 'w') as file:
                json.dump(results, file, default=str, indent=4)
            
            # Create visualizations
            self.create_visualizations()
            
            # Save basic statistics
            stats_path = os.path.join(self.results_directory, f'basic_statistics_{self.timestamp}.csv')
            basic_stats = self.dataframe.groupby('group')[self.categories + ['total_score']].agg(['mean', 'std', 'min', 'max'])
            basic_stats.to_csv(stats_path)
            
            print(f"\nAnalysis complete. Results saved in '{self.results_directory}' directory:")
            print(f"- Summary report: {report_path}")
            print(f"- Full results: {results_path}")
            print(f"- Basic statistics: {stats_path}")
            print("- Visualizations: Multiple PNG files")
            
            return results
            
        except Exception as error:
            error_results = {
                'error': str(error),
                'error_type': str(type(error).__name__),
                'traceback': str(error.__traceback__)
            }
            print(f"Error during analysis: {str(error)}")
            return error_results

if __name__ == "__main__":
    try:
        analyzer = CognitiveBehavioralTherapyAnalysis()
        results = analyzer.run_full_analysis()
        
        if 'error' in results:
            print("\nAnalysis failed with error:")
            print(f"Type: {results['error_type']}")
            print(f"Message: {results['error']}")
            raise Exception(results['error'])
            
    except Exception as error:
        print(f"\nFatal error during analysis: {str(error)}")
        raise
