#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 18:28:21 2018

@author: Soumya 
"""

# Environment where smart-keys agent learns to get better

'''
Environment performs the following tasks:
    1. Set the simulation ready to start.
    2. Execute the action decided by the learning algo, which in this scenario
       is to predict a word.
    3. Obtain the result -> is the prediction correct or not
    4. Compute a reward based on the prediction (Edit distance between the 
       actual word and the predicted word.)
    
Note:
    Agent takes only one action : Predicting the word 
    
    Observation definig state : Havent defined yet
'''

# Import ing Libraries

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

#from gym import error, spaces, utils


'''
TODO:
    State of the environment will be a list. Initially it will be empty and 
    at each step we add a word to the list.
    When we reach the end of the article the episode ends, _done is set to true
    and we start again from the beginning of the article.
'''

class KeysEnv(gym.Env):
    
    def read_text_file(filename):
        '''
        Function to read file and create generator
        
        Args:
            filename : Name/path of the file containing the text
            
        Return: 
            A string of all content in the text file
        '''
        with open(filename, 'r') as file:
            content = file.read()
            return content
        
    def __init__(self):
        '''
        Function for initializing the environment
        '''
        # No. of actions the agent can take, in this case agent only 
        # predicts next word
        self.action_space = spaces.Discrete(1)
        # Max and min values for rewards
        self.reward_range = (-np.inf,np.inf)
        # Setting seed for random number generator
        self._seed()
        
        # This list has all the words from the article sequentially 
        self.full_word_list = self.read_text_file("Sample_article.txt")
        
        # Environment state, initally empty, append a word at each step
        # This is what the agent observes
        self.state = []
        
        # Store a list of all the predicted words to display and compare
        self.predicted_word_history = []
        
        # Stores the target word for a particular iteration
        self.actual_word = ''
        
    def get_target_word(self):
        '''
        Function to return the actual word, primarily to be used as the target  
        label in the training loop.
        '''
        return self.actual_word
        
    def _seed(self, seed=None):
        '''
        Function initializes the random number generator
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    
    def _step(self, action):
        """
        Function is called at every step of RL
        
        Parameters
        ----------
        action : Action taken by the agent

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (list) :
                The state list now containing the actual words the agent has 
                seen so far. For the first iteration the state list os empty,
                words are appended to it in each iteration.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        
        '''
        NOTE:
            Figure out if the action is the prediction_word, / do we derive the 
            prediction word in the _take_action function as the only action the 
            agent takes is predicting one word.
        '''
        
        #predicted_word = self._take_action(action)
        predicted_word =''
        self.status = self.env.step()
        
        # Get the actual word (ground truth for that prediction)
        word_index = len(self.state)
        self.actual_word = self.full_word_list[word_index]
        
        # Append actual word to the state & predicted word to predicted list
        self.state.append(self.actual_word)
        self.predicted_word_history.append(predicted_word)
        
        # Calculate the reward, based on what is predicted
        reward = self._get_reward(self.actual_word, predicted_word)
        
        # Observation from the environment is the 
        ob = self.state
        
        # When all the words of the article have been seen the episode is over
        episode_over = (len(self.state) == len(self.full_word_list))

        return ob, reward, episode_over, {}
    
    def _reset(self):
        '''
        Function to reset the environment to its initial state
        '''
        self.full_word_list = self.read_text_file("Sample_article.txt")
        # Set state to inital state
        self.state = []
        
    def _render(self, mode='human', close=False):
        if close:
            return
        if mode is 'human':
            print(self.state)
            print(self.predicted_word_history)
            
    
    def _take_action(self, action):
        '''
        Should return the predicted_word
        
        Take self.state as input
        '''
        pass
    
    def calculate_edit_distance(actual_word, predicted_word):
        '''
        Funtion to find the Levenshtein Distance between two given strings.
        This gives us a measure of similarity between the two words.
        
        Args:
            actual_word : The actual word being written
            predicted_word : The word our agent predicted
        Return:
            The number of edits required to transform the predicted word
            to the actual word.
        '''
        # Create a table to store results of subproblems
        m = len(actual_word)
        n = len(predicted_word)
        dp = [[0 for x in range(n+1)] for x in range(m+1)]
        
        # Fill d[][] in bottom up manner 
        for i in range(m+1): 
            for j in range(n+1): 
      
                # If first string is empty, only option is to 
                # isnert all characters of second string 
                if i == 0: 
                    dp[i][j] = j    # Min. operations = j 
      
                # If second string is empty, only option is to 
                # remove all characters of second string 
                elif j == 0: 
                    dp[i][j] = i    # Min. operations = i 
      
                # If last characters are same, ignore last char 
                # and recur for remaining string 
                elif actual_word[i-1] == predicted_word[j-1]: 
                    dp[i][j] = dp[i-1][j-1] 
      
                # If last character are different, consider all 
                # possibilities and find minimum 
                else: 
                    dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                       dp[i-1][j],        # Remove 
                                       dp[i-1][j-1])      # Replace 
      
        return dp[m][n]

    def _get_reward(self, actual_word, predicted_word):
        """
        Function to calculate the reward for the agent based on the action
        
        Args:
            
        Returns:
            The reward earned by the agent
        """
        edit_distance = self.calculate_edit_distance(actual_word, 
                                                     predicted_word)
        
        # Assign reward based on the edit_distance
        # if distance between the strings is less then give higher reward
        # If the distance is 0 then the prediction is perfect. In such a case
        # give significantly higher reward, so the agent tries to claim it.
        # If the word requires more than 5 edits give negetive reward. 
        
        if edit_distance == 0:
            reward = 100
        elif edit_distance > 0 and edit_distance < 3 and len(actual_word) > 3:
            reward = 70
        elif edit_distance == 3 and len(actual_word) > 5:
            reward = 50
        elif edit_distance > 0 and edit_distance < 3 and len(actual_word) =< 3:
            reward = -1
        elif edit_distance == 3 and len(actual_word) <= 4:
            reward = -1
        else:
            reward = -1
            
        return reward
        
        
    
