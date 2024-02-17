import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class GraphQueryError(Exception):
    def __init__(self, message):
        self.message = message


def run(
        driver, 
        query, 
        params=None
        ):
    '''
    Run a Cypher query and return the results

    Args:
        driver: neo4j driver
        query: str, Cypher query
        params: dict, parameters for the query
    '''
    with driver.session() as session:
        if params is not None:
            return [r for r in session.run(query, params)]
        else:
            return [r for r in session.run(query)]

def degree_counts(
        gds, 
        node_label, 
        relationship_type, 
        direction='BOTH'
        ):
    '''
    Return the degree counts for a given node label and relationship type

    Args:
        gds: Graph Data Science driver
        node_label: str, label of the node
        relationship_type: str, type of the relationship
        direction: str, direction of the relationship, one of {'BOTH', 'IN', 'OUT'}
    '''
    dr = direction.upper()
    if dr not in {'BOTH', 'IN', 'OUT'}:
        raise GraphQueryError(f"direction must be one of {{'BOTH', 'IN', 'OUT'}}, but got {dr}")

    if dr == 'BOTH':
        pattern = f'[(n)-[:{relationship_type}]-() | n]'
    elif dr == 'OUT':
        pattern = f'[(n)-[:{relationship_type}]->() | n]'
    else:
        pattern = f'[(n)<-[:{relationship_type}]-() | n]'
    
    df= gds.run_cypher( f'''
            MATCH (n:{node_label}) WITH id(n) as nodeIds, size({pattern}) AS degree
            RETURN degree, count(degree) AS degreeCount ORDER BY degree
        ''')

    df['csum'] = df.degreeCount.cumsum()
    n = df.degreeCount.sum()
    df['percentile'] = df.csum/n
    return df.drop(columns=['csum'])

def get_percentiles(
        df, 
        q=None
        ):
    '''
    Return the percentiles of the degree distribution

    Args:
        df: pandas DataFrame, degree counts
        q: list, percentiles to calculate
    '''
    if q is None:
        q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    q_name = ['min'] + [f'p{int(100*i)}' for i in q] + ['max']
    p = [df.degree.min()] + [df.degree[df.percentile >= i].iloc[0] for i in q] + [df.degree.max()]
    p_df = pd.DataFrame(np.column_stack((q_name, p)), columns=['q', 'percentile'])
    return p_df.set_index('q')

def plot_nutritional_values(
        nutrVal
        ):
    '''
    Plot the distribution of nutritional values

    Args:
        nutrVal: pandas DataFrame, nutritional values
    '''
    f, axs = plt.subplots(1,5,figsize=(16,3))

    axs[0].plot(sorted(nutrVal['calories']))
    axs[0].set_yscale('log')
    axs[0].set_title('Calories Distribution')

    axs[1].plot(sorted(nutrVal['totalFat']))
    axs[1].set_yscale('log')
    axs[1].set_title('Total Fat Distribution')

    axs[2].plot(sorted(nutrVal['sugar']))
    axs[2].set_yscale('log')
    axs[2].set_title('Sugar Distribution')

    axs[3].plot(sorted(nutrVal['sodium']))
    axs[3].set_yscale('log')
    axs[3].set_title('Sodium Distribution')

    axs[4].plot(sorted(nutrVal['protein']))
    axs[4].set_yscale('log')
    axs[4].set_title('Protein Distribution')

    plt.show()

def get_nutritional_values(
        nutrVal,
        d_range=(25, 75),
        verbose=True
        ):
    '''
    Return the range of the nutritional values within the given percentile range

    Args:
        nutrVal: pandas DataFrame, nutritional values
        d_range: tuple, range of the percentiles
    '''
    calories_range = (
        np.percentile(nutrVal['calories'], d_range[0]), 
        np.percentile(nutrVal['calories'], d_range[1]))
    if verbose: 
        print(f"{d_range[1]-d_range[0]}% of calories data lies between " +
        f"{calories_range[0]:.2f} and {calories_range[1]:.2f}")

    fat_range = (
        np.percentile(nutrVal['totalFat'], d_range[0]), 
        np.percentile(nutrVal['totalFat'], d_range[1]))
    if verbose: 
        print(f"{d_range[1]-d_range[0]}% of total fat data lies between " +
        f"{fat_range[0]:.2f} and {fat_range[1]:.2f}")

    sugar_range = (
        np.percentile(nutrVal['sugar'], d_range[0]), 
        np.percentile(nutrVal['sugar'], d_range[1]))
    if verbose: 
        print(f"{d_range[1]-d_range[0]}% of sugar data lies between " +
        f"{sugar_range[0]:.2f} and {sugar_range[1]:.2f}")

    sodium_range = (
        np.percentile(nutrVal['sodium'], d_range[0]), 
        np.percentile(nutrVal['sodium'], d_range[1]))
    if verbose: 
        print(f"{d_range[1]-d_range[0]}% of sodium data lies between " +
        f"{sodium_range[0]:.2f} and {sodium_range[1]:.2f}")

    protein_range = (
        np.percentile(nutrVal['protein'], d_range[0]), 
        np.percentile(nutrVal['protein'], d_range[1]))
    if verbose: 
        print(f"{d_range[1]-d_range[0]}% of protein data lies between " +
        f"{protein_range[0]:.2f} and {protein_range[1]:.2f}")
    
    return calories_range, fat_range, sugar_range, sodium_range, protein_range

def plot_interactions_distribution(
        gds, 
        n_rew=100
        ):
    '''
    Plot the distribution of user interactions and reviews

    Args:
        gds: Graph Data Science driver
        n_rew: int, number of top interactions to plot
    '''
    all_interactions_df = degree_counts(gds, 'User', 'REVIEWED|SUBMITTED', 'OUT')
    reviews_df = degree_counts(gds, 'User', 'REVIEWED', 'OUT')

    f, axs = plt.subplots(1,2,figsize=(16,5))

    axs[0].bar(all_interactions_df.degree[:n_rew], all_interactions_df.degreeCount[:n_rew], width=1, log=True)
    axs[0].set_title('User Total Interactions Distribution')
    axs[0].set_ylabel('User Count')
    axs[0].set_xlabel('Number of Total Interactions')
    plt.figtext(0.4, 0.5, get_percentiles(all_interactions_df).to_string())

    axs[1].bar(reviews_df.degree[:n_rew], reviews_df.degreeCount[:n_rew], width=1, log=True)
    axs[1].set_title('User Reviews Distribution')
    axs[1].set_ylabel('User Count')
    axs[1].set_xlabel('Number of Reviews')
    plt.figtext(0.83, 0.5, get_percentiles(reviews_df).to_string())

    plt.show()


def plot_recipe_interactions_distribution(
        gds, 
        n_rew=100
        ):
    '''
    Plot the distribution of recipe interactions and reviews

    Args:
        gds: Graph Data Science driver
        n_rew: int, number of top interactions to plot
    '''
    all_interactions_df = degree_counts(gds, 'Recipe', 'REVIEWED|SUBMITTED', 'IN')
    reviews_df = degree_counts(gds, 'Recipe', 'REVIEWED', 'IN')
    submit_df = degree_counts(gds, 'Recipe', 'SUBMITTED', 'IN')

    f, axs = plt.subplots(1,2,figsize=(16,5))

    axs[0].bar(all_interactions_df.degree[:n_rew], all_interactions_df.degreeCount[:n_rew], width=1, log=True)
    axs[0].set_title('Recipe Total Interactions Distribution')
    axs[0].set_ylabel('Recipe Count')
    axs[0].set_xlabel('Number of Total Interactions')
    plt.figtext(0.4, 0.5, get_percentiles(all_interactions_df).to_string())


    axs[1].bar(reviews_df.degree[:n_rew], reviews_df.degreeCount[:n_rew], width=1, log=True)
    axs[1].set_title('Recipe Reviews Distribution')
    axs[1].set_ylabel('Recipe Count')
    axs[1].set_xlabel('Number of Reviews')
    plt.figtext(0.83, 0.5, get_percentiles(reviews_df).to_string())

    plt.show()