import numpy as np
from collections import deque


class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        return advantages

    def update_get_highest_and_lowest(self, prompts, rewards, exp=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0
        highest_index = np.empty_like(rewards, dtype=int) * 0
        lowest_index = np.empty_like(rewards, dtype=int) * 0
        
        # 记录每个样本的原始索引
        original_indices = np.arange(len(rewards))
        
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            
            # 获取当前prompt对应的mask和原始索引
            mask = prompts == prompt
            prompt_rewards = rewards[mask]  # Fix: Recalculate prompt_rewards for each prompt
            prompt_indices = original_indices[mask]

            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            advantages[mask] = (prompt_rewards - mean) / std
            
            # 找到组内最大和最小reward的索引
            if len(prompt_rewards) > 0:
                max_index_in_group = np.argmax(prompt_rewards)
                min_index_in_group = np.argmin(prompt_rewards)
                
                # 转换为全局索引
                global_max_index = prompt_indices[max_index_in_group]
                global_min_index = prompt_indices[min_index_in_group]
                
                # 设置索引数组：组内所有元素都填充组内最大reward的全局索引
                highest_index[mask] = global_max_index
                # 组内所有元素都填充组内最小reward的全局索引
                lowest_index[mask] = global_min_index
        
        return advantages, highest_index, lowest_index 

    def update_frame(self, prompts, rewards, exp=False):
        """
        prompts: List[str] length B
        rewards: np.ndarray shape [B, F]
        """

        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)

        B, F = rewards.shape

        unique = np.unique(prompts)

        advantages = np.zeros_like(rewards)

        for prompt in unique:

            mask = prompts == prompt
            prompt_rewards = rewards[mask]  # shape [n_prompt, F]

            if prompt not in self.stats:
                self.stats[prompt] = []

            self.stats[prompt].append(prompt_rewards)

            self.history_prompts.add(hash(prompt))

        for prompt in unique:

            # history rewards
            self.stats[prompt] = np.concatenate(self.stats[prompt], axis=0)  # [N_history, F]

            mask = prompts == prompt
            prompt_rewards = rewards[mask]  # [n_prompt, F]

            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)  # [1, F]

            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # [1, F]
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4

            advantages[mask] = (prompt_rewards - mean) / std

        return advantages

    def update_inner_group(self, prompts, rewards, exp=False):
        """
        prompts: (L,) - each entry is a prompt identifier (can be repeated, not continuous)
        rewards: (L, inner_group_size)
        """
        prompts = np.asarray(prompts)
        rewards = np.asarray(rewards, dtype=np.float64)

        L, G = rewards.shape
        advantages = np.zeros_like(rewards)

        unique_prompts = np.unique(prompts)

        for prompt in unique_prompts:
            # 找到所有属于该 prompt 的样本
            idx = np.where(prompts == prompt)[0]  # array of positions (not necessarily continuous)
            prompt_rewards = rewards[idx]         # shape: (num_samples_for_prompt, G)
            flattened = prompt_rewards.reshape(-1)  # flatten K×G

            # 统计 mean/std
            mean = flattened.mean()
            if self.global_std:
                std = rewards.reshape(-1).std() + 1e-4
            else:
                std = flattened.std() + 1e-4

            # 计算 advantage 并放回原位置
            advantages[idx] = (prompt_rewards - mean) / std

        return advantages
    
    def update_inner_group_rank(self, prompts, rewards):
        """
        prompts: (L,) - each entry is a prompt identifier (can be repeated, not continuous)
        rewards: (L, frame)
        """
        prompts = np.asarray(prompts)
        rewards = np.asarray(rewards, dtype=np.float64)

        L, G = rewards.shape
        advantages = np.zeros_like(rewards)

        unique_prompts = np.unique(prompts)

        for prompt in unique_prompts:
            # 找到所有属于该 prompt 的样本
            idx = np.where(prompts == prompt)[0]  # positions of this prompt
            prompt_rewards = rewards[idx]         # shape: (num_samples_for_prompt, G)
            flattened = prompt_rewards.reshape(-1)  # flatten K×G

            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].append(flattened.copy())
            self.history_prompts.add(hash(prompt)) 
            
            # 统计 mean/std
            mean = flattened.mean()
            if self.global_std:
                std = rewards.reshape(-1).std() + 1e-4
            else:
                std = flattened.std() + 1e-4

            # 计算 advantage 并放回原位置
            advantages[idx] = (prompt_rewards - mean) / std

        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}

    def get_mean_of_top_rewards(self, top_percentage):
        if not self.stats:
            return 0.0

        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            if isinstance(prompt_rewards, list):
                rewards = np.array(prompt_rewards)
            else:
                rewards = prompt_rewards

            if rewards.size == 0:
                continue

            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue

            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)

            top_rewards = rewards[rewards >= threshold]

            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        if not per_prompt_top_means:
            return 0.0

        return np.mean(per_prompt_top_means)


# def main():
#     tracker = PerPromptStatTracker()
#     prompts = ["a", "b", "a", "c", "b", "a"]
#     rewards = [1, 2, 3, 4, 5, 6]
#     advantages = tracker.update(prompts, rewards)
#     print("Advantages:", advantages)
#     avg_group_size, history_prompts = tracker.get_stats()
#     print("Average Group Size:", avg_group_size)
#     print("History Prompts:", history_prompts)
#     tracker.clear()
#     print("Stats after clear:", tracker.stats)


# if __name__ == "__main__":
#     main()

def main():
    tracker = PerPromptStatTracker()
    
    # 假设 inner_group_size = 2
    prompts = ["a", "b", "a", "c", "b", "a"]
    rewards = np.array([
        [1, 2],
        [2, 3],
        [3, 1],
        [4, 5],
        [5, 4],
        [6, 0]
    ])  # shape (6,2)
    
    advantages = tracker.update_inner_group_rank(prompts, rewards)
    print("Advantages:\n", advantages)
    
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    
    tracker.clear()
    print("Stats after clear:", tracker.stats)


if __name__ == "__main__":
    main()
