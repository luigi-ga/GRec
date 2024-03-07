import time
import numpy as np

def get_node_counts(
        gds,
        verbose=True
        ):
    '''
    Returns the total number of nodes in the graph

    Args:
        gds: Graph Data Science driver
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher( '''
        CALL apoc.meta.stats()
        YIELD labels AS nodeCounts
        UNWIND keys(nodeCounts) AS label
        WITH label, nodeCounts[label] AS nodeCount
        WHERE label IN ['User','Recipe']
        RETURN label, nodeCount
    ''')
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def get_relationship_counts(
        gds,
        verbose=True
        ):
    '''
    Returns the total number of relationships in the graph

    Args:
        gds: Graph Data Science driver
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher( '''
        CALL apoc.meta.stats()
        YIELD relTypesCount as relationshipCounts
        UNWIND keys(relationshipCounts) AS type
        WITH type, relationshipCounts[type] AS relationshipCount
        WHERE type IN ['REVIEWED','SUBMITTED']
        RETURN type, relationshipCount
    ''')
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def get_random_user(
        gds, 
        n_rew=100,
        verbose=True
        ):
    '''
    Returns a random user that has reviewed at least n_rew recipes

    Args:
        gds: Graph Data Science driver
        n_rew: int, minimum number of reviews
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher('''
        MATCH (u:User)-[:REVIEWED]->(:Recipe)
        WITH u, COUNT(*) AS reviewCount
        WHERE reviewCount >= $n_reviews
        WITH u, RAND() AS randomOrder, reviewCount
        ORDER BY randomOrder
        RETURN u.id AS randomUserID, reviewCount
        LIMIT 1
    ''', params={'n_reviews': n_rew})
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def get_user_interactions(
        gds, 
        user_id,
        verbose=True
        ):
    '''
    Returns all the interactions (submissions and reviews) of a given user

    Args:
        gds: Graph Data Science driver
        user_id: str, user identifier
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher('''
        MATCH (u:User {id: $userId})-[i:SUBMITTED|REVIEWED]->(r:Recipe)
        RETURN r.id AS recipeID,
            r.name AS name,
            r.nutrition AS nutrition,
            r.n_ingredients AS n_ingredients,
            COALESCE(i.submitted, i.date) AS interactionDate,
            type(i) AS interactionType
        ORDER BY interactionType, interactionDate DESC
        ''', params={'userId': user_id})
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def get_hops_count(
        gds, 
        user_id,
        verbose=True
        ):
    '''
    Returns the number of nodes we would need to traverse over to get the
    recommendations for the collaborative filtering approach

    Args:
        gds: Graph Data Science driver
        user_id: str, user identifier
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher('''
        MATCH (u1:User {id: $userId})-[r:REVIEWED|SUBMITTED]->(recipe:Recipe)
        WITH u1, COLLECT(recipe) AS userRecipes
        MATCH (u1)-[:REVIEWED]->(r1:Recipe)<-[:REVIEWED|SUBMITTED]-(u2:User)-[:REVIEWED|SUBMITTED]->(r2:Recipe)
        WHERE NOT r2 IN userRecipes
        RETURN u1.id AS userId,
            COUNT(DISTINCT r1) AS interactedRecipesCount,
            COUNT(DISTINCT u2) AS likeUsersCount,
            COUNT(DISTINCT r2) AS potentialRecommendationsCount
        ''', params={
            'userId': user_id})
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def get_user_favourite_ingredients(
        gds, 
        user_id, 
        excluded_ingr,
        percentil=50,
        verbose=True
        ):
    '''
    Returns the top max_fav_ingr of an user excluding excluded_ingr.
    User favorite ingredients are extracted from the recipes the user
    has published and from the recipe the user evaluated with 5 rating.

    Args:
        gds: Graph Data Science driver
        user_id: str, user identifier
        max_fav_ingr: int, number of favorite ingredients to return
        excluded_ingr: list, list of excluded ingredients
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher('''                                                  
        MATCH (u:User {id: $userId})-[rel:REVIEWED|SUBMITTED]->(r:Recipe)-[:WITH_INGREDIENTS]->(i:Ingredient)
        WHERE (rel.rating >= 4 OR TYPE(rel) = 'SUBMITTED') AND NOT i.name IN $excluded_ingr
        WITH i, COUNT(r) AS favCount, COLLECT(r.id) AS favRecipes
        ORDER BY favCount DESC
        RETURN i.name AS favoriteIngredient, favCount, favRecipes
        ''', params={
            'userId': user_id, 
            'excluded_ingr': excluded_ingr})
    if not query.empty:
        # Calculate the threshold count for the given percentile
        # To do so, we need to remove duplicates and calculate the percentile
        threshold_count = np.percentile(list(set(query['favCount'])), percentil)
        # Filter the DataFrame for ingredients above the threshold
        query = query[query['favCount'] >= threshold_count]
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def get_recipe_w_ingreds(
        gds, 
        excluded_recipes, 
        favorite_ingredients, 
        verbose=True
        ):
    '''
    Returns recipies that have the same ingredients as favorite_ingredients, 
    excluding from those excluded_recipes and limiting the output to n_suggestions

    Args:
        gds: Graph Data Science driver
        n_suggestions: int, number of suggestions to return
        excluded_recipes: list, list of recipes to be excluded
        favorite_ingredients: list, list of favorite ingredients
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher('''
        MATCH (r:Recipe)-[:WITH_INGREDIENTS]->(i:Ingredient)
        WHERE NOT r.id IN $excludedRecipes
        WITH r, COLLECT(i.name) AS recipeIngredients
        WHERE ANY(ingredient IN recipeIngredients WHERE ingredient IN $favIngreds)
        RETURN r.id AS recipeID, r.name AS recipeName, recipeIngredients,
            SIZE([ingredient IN recipeIngredients WHERE ingredient IN $favIngreds]) AS matchCount,
            SIZE(recipeIngredients) AS totalIngredients,
            (toFloat(SIZE([ingredient IN recipeIngredients WHERE ingredient IN $favIngreds])) / SIZE(recipeIngredients) * log(1 + SIZE(recipeIngredients))) AS relevanceScore
        ORDER BY relevanceScore DESC
        ''', params={
            'excludedRecipes': excluded_recipes,
            'favIngreds': favorite_ingredients})
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def get_user_nutritional_values(
        gds, 
        user_id, 
        verbose=True
        ):
    '''
    Returns the nutritional values of the recipes that the user has interacted with

    Args:
        gds: Graph Data Science driver
        user_id: str, user identifier
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher('''
        MATCH (u:User {id:$userID})-[rel:SUBMITTED|REVIEWED]->(r:Recipe)
        WHERE (rel.rating >= 4 OR TYPE(rel) = 'SUBMITTED')
        WITH r, apoc.convert.fromJsonList(r.nutrition) AS nutritionList
        RETURN  r.id AS recipeID,
                nutritionList[0] AS calories, 
                nutritionList[1] AS totalFat,
                nutritionList[2] AS sugar,
                nutritionList[3] AS sodium,
                nutritionList[4] AS protein,
                nutritionList[5] AS saturatedFat,
                nutritionList[6] AS carbs
        ''', params={
            'userID': user_id})
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def get_recipe_nutritional_values(
        gds, 
        interacted_recipes, 
        calories_range, 
        fat_range, 
        sugar_range, 
        sodium_range, 
        protein_range,
        sat_fat_range, 
        carbs_range,
        verbose=True
        ):
    '''
    Returns recipies that have the same nutritional values as the given ranges,
    excluding from those interacted_recipes and limiting the output to n_suggestions

    Args:
        gds: Graph Data Science driver
        n_suggestions: int, number of suggestions to return
        interacted_recipes: list, list of recipes the user has interacted with
        calories_range: tuple, range of calories
        fat_range: tuple, range of fat
        sugar_range: tuple, range of sugar
        sodium_range: tuple, range of sodium
        protein_range: tuple, range of protein
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher('''
    MATCH (r:Recipe)
    WHERE NOT r.id IN $interactedRecipes
    WITH r, apoc.convert.fromJsonList(r.nutrition) AS nutritionList
    WHERE   nutritionList[0] > $min_0 AND nutritionList[0] < $max_0    // CALORIES
        AND nutritionList[1] > $min_1 AND nutritionList[1] < $max_1    // TOTAL FAT (PDV)
        AND nutritionList[2] > $min_1 AND nutritionList[2] < $max_2    // SUGAR (PVD)
        AND nutritionList[3] > $min_3 AND nutritionList[3] < $max_3    // SODIUM (PDV)
        AND nutritionList[4] > $min_4 AND nutritionList[4] < $max_4    // PROTEIN
        AND nutritionList[5] > $min_5 AND nutritionList[5] < $max_5    // SATURATED FAT (PDV)
        AND nutritionList[6] > $min_6 AND nutritionList[6] < $max_6    // CARBS
    RETURN r.id as recipeID, r.name AS recipeName, nutritionList
    ''', params={ 
        'interactedRecipes': interacted_recipes,
        'min_0': calories_range[0], 'max_0': calories_range[1],
        'min_1': fat_range[0], 'max_1': fat_range[1],
        'min_2': sugar_range[0], 'max_2': sugar_range[1],
        'min_3': sodium_range[0], 'max_3': sodium_range[1],
        'min_4': protein_range[0], 'max_4': protein_range[1],
        'min_5': sat_fat_range[0], 'max_5': sat_fat_range[1],
        'min_6': carbs_range[0], 'max_6': carbs_range[1]})
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def get_recipe_nutritional_ingreds(
        gds, 
        interacted_recipes, 
        favorite_ingredients,
        calories_range, 
        fat_range, 
        sugar_range, 
        sodium_range, 
        protein_range,
        sat_fat_range,
        carbs_range,
        verbose=True
        ):
    '''
    Returns recipies that have the same nutritional values as the given ranges
    and ingredients as favorite_ingredients, excluding from those interacted_recipes
    and limiting the output to n_suggestions. The output is ordered by the number of
    matching ingredients with the favorite ones.

    Args:
        gds: Graph Data Science driver
        n_suggestions: int, number of suggestions to return
        interacted_recipes: list, list of recipes the user has interacted with
        favorite_ingredients: list, list of favorite ingredients
        calories_range: tuple, range of calories
        fat_range: tuple, range of fat
        sugar_range: tuple, range of sugar
        sodium_range: tuple, range of sodium
        protein_range: tuple, range of protein
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher('''
    MATCH (r:Recipe)-[:WITH_INGREDIENTS]->(i:Ingredient)
    WHERE NOT r.id IN $interactedRecipes
    WITH r, COLLECT(i.name) AS recipeIngredients, apoc.convert.fromJsonList(r.nutrition) AS nutritionList
    WHERE 
        ANY(ingredient IN recipeIngredients WHERE ingredient IN $favIngreds) 
        AND nutritionList[0] > $min_0 AND nutritionList[0] < $max_0    // CALORIES
        AND nutritionList[1] > $min_1 AND nutritionList[1] < $max_1    // TOTAL FAT (PDV)
        AND nutritionList[2] > $min_1 AND nutritionList[2] < $max_2    // SUGAR (PVD)
        AND nutritionList[3] > $min_3 AND nutritionList[3] < $max_3    // SODIUM (PDV)
        AND nutritionList[4] > $min_4 AND nutritionList[4] < $max_4    // PROTEIN
        AND nutritionList[5] > $min_5 AND nutritionList[5] < $max_5    // SATURATED FAT (PDV)
        AND nutritionList[6] > $min_6 AND nutritionList[6] < $max_6    // CARBS
    RETURN  r.id as recipeID, r.name AS recipeName, recipeIngredients, nutritionList,
            (toFloat(SIZE([ingredient IN recipeIngredients WHERE ingredient IN $favIngreds])) / SIZE(recipeIngredients) * log(1 + SIZE(recipeIngredients))) AS relevanceScore
    ORDER BY relevanceScore DESC
    ''', params={
        'interactedRecipes': interacted_recipes,
        'favIngreds': favorite_ingredients,
        'min_0': calories_range[0], 'max_0': calories_range[1],
        'min_1': fat_range[0], 'max_1': fat_range[1],
        'min_2': sugar_range[0], 'max_2': sugar_range[1],
        'min_3': sodium_range[0], 'max_3': sodium_range[1],
        'min_4': protein_range[0], 'max_4': protein_range[1],
        'min_5': sat_fat_range[0], 'max_5': sat_fat_range[1],
        'min_6': carbs_range[0], 'max_6': carbs_range[1]})
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def get_recipes_taglist(
        gds,
        verbose=True
        ):
    '''
    Returns the taglist of all recipes with count

    Args:  
        gds: Graph Data Science driver
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher('''
        MATCH (r:Recipe)
        WITH r, apoc.convert.fromJsonList(r.tags) AS tagList
        UNWIND tagList AS tag
        RETURN tag, COUNT(*) AS tagCount
        ORDER BY tagCount DESC
        ''')
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def get_user_top_tags(
        gds, 
        user_id, 
        excluded_tags, 
        percentil=50,
        verbose=True
        ):
    '''
    Returns the top max_tags tags of an user excluding excluded_tags

    Args:
        gds: Graph Data Science driver
        user_id: str, user identifier
        excluded_tags: list, list of excluded tags
        max_tags: int, number of tags to return
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher('''
        MATCH (u:User {id:$user_id})-[rel:SUBMITTED|REVIEWED]->(r:Recipe)
        WHERE (rel.rating >= 4 OR TYPE(rel) = 'SUBMITTED')
        WITH r, apoc.convert.fromJsonList(r.tags) AS tagList
        UNWIND tagList AS tag
        WITH tag
        WHERE NOT tag IN $excluded_tags
        RETURN tag, COUNT(*) AS tagCount
        ORDER BY tagCount DESC
        ''', params={
            'user_id': user_id, 
            'excluded_tags': excluded_tags})
    if not query.empty:
        # Calculate the threshold count for the given percentile
        # To do so, we need to remove duplicates and calculate the percentile
        threshold_count = np.percentile(list(set(query['tagCount'])), percentil)
        # Filter the DataFrame for tags count above the threshold
        query = query[query['tagCount'] >= threshold_count]
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def find_top_tag_matching_recipes(
        gds, 
        interacted_recipes, 
        top_tags,
        verbose=True
        ):
    '''
    Returns recipies that have the same tags as top_tags, excluding from those
    interacted_recipes and limiting the output to n_suggestions

    Args:
        gds: Graph Data Science driver
        n_suggestions: int, number of suggestions to return
        interacted_recipes: list, list of recipes the user has interacted with
        top_tags: list, list of top tags
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher('''
        MATCH (r:Recipe)
        WHERE NOT r.id IN $interactedRecipes
        WITH r, apoc.convert.fromJsonList(r.tags) AS tagList
        WHERE ANY(tag IN tagList WHERE tag IN $topTags)
        RETURN r.id AS recipeID, r.name AS recipeName, tagList,
            SIZE([tag IN tagList WHERE tag IN $topTags]) AS matchCount,
            SIZE(tagList) AS totalTags,
            (toFloat(SIZE([tag IN tagList WHERE tag IN $topTags])) / SIZE(tagList) * log(1 + SIZE(tagList))) AS relevanceScore
        ORDER BY relevanceScore DESC
        ''', params={ 
            'interactedRecipes': interacted_recipes,
            'topTags': top_tags})
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query

def find_matching_recipes_with_nutrition_and_tags(
        gds, 
        interacted_recipes, 
        favorite_ingredients, 
        top_tags, 
        calories_range, 
        fat_range, 
        sugar_range, 
        sodium_range, 
        protein_range,
        sat_fat_range,
        carbs_range,
        verbose=True
        ):
    '''
    Returns recipies that have the same tags as top_tags and ingredients as
    favorite_ingredients, excluding from those interacted_recipes and limiting
    the output to n_suggestions

    Args:
        gds: Graph Data Science driver
        n_suggestions: int, number of suggestions to return
        interacted_recipes: list, list of recipes the user has interacted with
        favorite_ingredients: list, list of favorite ingredients
        top_tags: list, list of top tags
        calories_range: tuple, range of calories
        fat_range: tuple, range of fat
        sugar_range: tuple, range of sugar
        sodium_range: tuple, range of sodium
        protein_range: tuple, range of protein
        verbose: bool, print the execution time
    '''
    start_time = time.time()
    query = gds.run_cypher('''
        MATCH (r:Recipe)-[:WITH_INGREDIENTS]->(i:Ingredient)
        WHERE NOT r.id IN $interactedRecipes
        WITH r, COLLECT(i.name) AS recipeIngredients, apoc.convert.fromJsonList(r.nutrition) AS nutritionList, apoc.convert.fromJsonList(r.tags) AS tagList
        WHERE ANY(ingredient IN recipeIngredients WHERE ingredient IN $favIngreds)
            AND nutritionList[0] > $min_0 AND nutritionList[0] < $max_0    // CALORIES
            AND nutritionList[1] > $min_1 AND nutritionList[1] < $max_1    // TOTAL FAT (PDV)
            AND nutritionList[2] > $min_1 AND nutritionList[2] < $max_2    // SUGAR (PVD)
            AND nutritionList[3] > $min_3 AND nutritionList[3] < $max_3    // SODIUM (PDV)
            AND nutritionList[4] > $min_4 AND nutritionList[4] < $max_4    // PROTEIN
            AND nutritionList[5] > $min_5 AND nutritionList[5] < $max_5    // SATURATED FAT (PDV)
            AND nutritionList[6] > $min_6 AND nutritionList[6] < $max_6    // CARBS
            AND ANY(tag IN tagList WHERE tag IN $topTags)
        RETURN r.id AS recipeID, r.name AS recipeName,
            SIZE([ingredient IN recipeIngredients WHERE ingredient IN $favIngreds]) AS matchingIngreds,
            SIZE([tag IN tagList WHERE tag IN $topTags]) AS matchingTags, 
            (toFloat(SIZE([ingredient IN recipeIngredients WHERE ingredient IN $favIngreds])) / SIZE(recipeIngredients) * log(1 + SIZE(recipeIngredients))) AS ingrRelScore,
            (toFloat(SIZE([tag IN tagList WHERE tag IN $topTags])) / SIZE(tagList) * log(1 + SIZE(tagList))) AS tagRelScore
        ORDER BY (ingrRelScore + tagRelScore) DESC
        ''', params={
            'favIngreds': favorite_ingredients,
            'interactedRecipes': interacted_recipes,
            'topTags': top_tags,
            'min_0': calories_range[0], 'max_0': calories_range[1],
            'min_1': fat_range[0], 'max_1': fat_range[1],
            'min_2': sugar_range[0], 'max_2': sugar_range[1],
            'min_3': sodium_range[0], 'max_3': sodium_range[1],
            'min_4': protein_range[0], 'max_4': protein_range[1],
            'min_5': sat_fat_range[0], 'max_5': sat_fat_range[1],
            'min_6': carbs_range[0], 'max_6': carbs_range[1]})
    end_time = time.time()
    if verbose: print(f'\nQuery executed in {end_time-start_time:.2f} seconds')
    return query