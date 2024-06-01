

## In online_update.py in exploration_mod:

    def update_cliff_before(self, key, data: GridData, learning_rate: float, s_type: str = "sEM") -> List:
        print("Start updating the clusters.")
        before_cliff = data.cliff
        print("Before: ", before_cliff)

        if len(before_cliff) == 0:
            cliffs = self.build_cliff(key, data)
            return cliffs
        else:
            raw_data = np.array(data.new_data)
            
        num_observations = len(raw_data)
        
        
        
        wind_num = int(self.config_params["wind_num"])
        wind_k = np.arange(-wind_num, wind_num + 1)
        cluster_nums = len(before_cliff)
        
        ## m: [velocity, motion_angle]
        m = np.array([row[2:4] for row in before_cliff])
        c = np.array([[[row[4], row[5]], [row[6], row[7]]] for row in before_cliff])
        p = np.array([row[8] for row in before_cliff])


        r_batch = np.zeros((cluster_nums, len(wind_k), num_observations), dtype=float)

        for j in range(cluster_nums):
            for k in range(len(wind_k)):
                wrap_num = wind_k[k]
                try:
                    likelihood = np.array([multivariate_normal.pdf(np.array([row[0], row[1] + 2 * np.pi * wrap_num ]), mean=m[j,:], cov=c[j,:,:], allow_singular=True) for row in raw_data]) * p[j]
                except:
                    likelihood = 1e-9
                    # print("################################---------------------")
                    # print(likelihood)
                    # c[j,:,:] += np.eye(c[j,:,:].shape[0]) * 1e-6
                    # print(c[j,:,:])
                    # likelihood = np.array([multivariate_normal.pdf(np.array([row[0], row[1] + 2 * np.pi * wrap_num ]), mean=m[j,:], cov=c[j,:,:], allow_singular=True) for row in raw_data]) * p[j]
                    # print(likelihood)
                r_batch[j,k,:] = likelihood
        r_batch[r_batch < np.finfo(float).eps] = 0
        
        if np.sum(r_batch) < 1e-1:
            return False
        
        sum_r = np.tile(np.sum(r_batch, axis=(0, 1)), (cluster_nums, len(wind_k), 1))
        r_batch = np.divide(r_batch, sum_r, out=np.zeros_like(r_batch), where=sum_r!=0)
        
        m_new = np.zeros((cluster_nums, 2), dtype=float)
        for j in range(cluster_nums):
            t = np.zeros((num_observations, 2), dtype=float)
            for k in range(len(wind_k)):
                wrap_num = wind_k[k]
                data_copy = raw_data.copy()
                data_copy[:, 1] += 2 * np.pi * wrap_num
                t += data_copy * np.tile(r_batch[j,k,:].reshape(-1,1), (1, 2))

            sum_r_j = np.sum(r_batch[j,:,:])
            m_new[j,:] = np.divide(np.sum(t, axis=0), sum_r_j, where=sum_r_j!=0)
            if sum_r_j == 0:
                m_new[j,:] = np.zeros_like(m_new[j,:])
                
        c_new = np.zeros((cluster_nums,2,2), dtype=float)
        for j in range(cluster_nums):
            t = np.zeros((num_observations, 2, 2), dtype=float)
            for k in range(len(wind_k)):
                wrap_num = wind_k[k]
                data_copy = raw_data.copy()
                data_copy[:, 1] += 2 * np.pi * wrap_num
                d_mod = data_copy - np.tile(m_new[j,:], (num_observations, 1))
                t[:,0,0] += d_mod[:,0]**2 * r_batch[j,k,:]
                t[:,1,1] += d_mod[:,1]**2 * r_batch[j,k,:]
                t[:,0,1] += d_mod[:,0] * d_mod[:, 1] * r_batch[j,k,:]
                t[:,1,0] = t[:,0,1]
                
            sum_r_j = np.sum(r_batch[j,:,:])
            c_new[j,:,:] = np.divide(np.sum(t, axis=0), sum_r_j, where=sum_r_j!=0)
            if sum_r_j == 0:
                c_new[j,:,:] = np.zeros_like(c_new[j,:,:])

        p_new = np.ones((cluster_nums)) * (1 / cluster_nums)
        for j in range(cluster_nums):
            p_new[j] = np.sum(r_batch[j,:,:]) / num_observations


        # m_updated = np.zeros((cluster_nums, 2), dtype=float)
        # for i in range(cluster_nums):
        #     m_updated[i, 0] = m[i, 0] + learning_rate * (m_new[i, 0] - m[i, 0])
        #     phi_diff = np.arctan2(np.sin(m_new[i, 1] - m[i, 1]), np.cos(m_new[i, 1] - m[i, 1]))
        #     m_updated[i, 1] = m[i, 1] + learning_rate * phi_diff
        #     m_updated[i, 1] = utils.wrap_to_2pi_no_round(m[i, 1])
        
        # print("---------------1")
        # print(m_new, c_new, p_new)
        
        m = m + learning_rate * (m_new - m)
        c = c + learning_rate * (c_new - c)
        p = p + learning_rate * (p_new - p)
        
        # print("---------------2")
        # print(m, c, p)
        
        
        # ################################## M-step ##################################

        for j in range(cluster_nums):
            try:
                np.linalg.cholesky(c[j,:,:])
                chol_f = 0
            except np.linalg.LinAlgError:
                chol_f = 1
            if (chol_f != 0 and c[j,0,0] > 10**(-10) and c[j,1,1] > 10**(-10)) or (np.linalg.cond(c[j,:,:]) > 1/10**(-10)):
                c[j,:,:] += np.eye(2) * 10**(-10)

        ## discarding clusters with too small covariance
        self.RemNan = 0
        self.Remsmal = 0
        new_cluster_nums = cluster_nums

        rem = np.zeros(cluster_nums, dtype=float)
        rs = 0
        rn = 0
        for j in range(cluster_nums):
            if c[j,0,0] < 10**(-8) or c[j,1,1] < 10**(-8):
                rem[j] = 1
                new_cluster_nums -= 1
                self.Remsmal += 1
                rs += 1
            if np.linalg.cond(c[j,:,:]) > 1/10**(-10):
                rem[j] = 1
                new_cluster_nums -= 1
                self.RemNan += 1
                rn += 1

        rem = rem.astype(bool)
        
        if new_cluster_nums < cluster_nums:
            c = c[~rem,:,:]
            m = m[~rem,:]
            p = p[~rem]
            cluster_nums = new_cluster_nums
            p = p / np.sum(p)

        if m.size == 0:
            return
        
        m[:,1] = utils.wrap_to_2pi_no_round(m[:,1])

        # print("---------------4")
        # print(m,c,p)

        cliffs = []
        for cluster_i in range(len(m)):
            ### change order saving to the same as running code, 
            save_row = [
                key[0], key[1],
                m[cluster_i,0], m[cluster_i,1],
                c[cluster_i,0,0], c[cluster_i,0,1], c[cluster_i,1,0], c[cluster_i,1,1],
                p[cluster_i], None
            ]

            rounded_save_row = [round(value, 5) if not (value is None) else value for value in save_row]
            # utils.save_cliff_csv(self.cliff_csv_file, rounded_save_row)
            # print("Update with new data: ")
            # print(rounded_save_row)
            cliffs.append(rounded_save_row)
        
        
        return cliffs
    
    
            ############# here is the part for the original code, now put this part out and in a separate function compute_sufficient_statistics ################
        # r_batch = np.zeros((cluster_nums, len(wind_k), num_observations), dtype=float)

        # for j in range(cluster_nums):
        #     for k in range(len(wind_k)):
        #         wrap_num = wind_k[k]
        #         try:
        #             likelihood = np.array([multivariate_normal.pdf(np.array([row[0], row[1] + 2 * np.pi * wrap_num ]), mean=m[j,:], cov=c[j,:,:], allow_singular=True) for row in raw_data]) * p[j]
        #         except:
        #             likelihood = 1e-9
        #             # print("################################---------------------")
        #             # print(likelihood)
        #             # c[j,:,:] += np.eye(c[j,:,:].shape[0]) * 1e-6
        #             # print(c[j,:,:])
        #             # likelihood = np.array([multivariate_normal.pdf(np.array([row[0], row[1] + 2 * np.pi * wrap_num ]), mean=m[j,:], cov=c[j,:,:], allow_singular=True) for row in raw_data]) * p[j]
        #             # print(likelihood)
        #         r_batch[j,k,:] = likelihood
        # r_batch[r_batch < np.finfo(float).eps] = 0
        
        # if np.sum(r_batch) < 1e-1:
        #     return False
        
        # sum_r = np.tile(np.sum(r_batch, axis=(0, 1)), (cluster_nums, len(wind_k), 1))
        # r_batch = np.divide(r_batch, sum_r, out=np.zeros_like(r_batch), where=sum_r!=0)
        

        # ### N_k
        # N_new = np.zeros((cluster_nums, 1), dtype=float)
        # for j in range(cluster_nums):
        #     sum_r_j = np.sum(r_batch[j,:,:])
        #     N_new[j,:] = sum_r_j


        # ### S_k
        # S_new = np.zeros((cluster_nums, 2), dtype=float)
        # for j in range(cluster_nums):
        #     t = np.zeros((num_observations, 2), dtype=float)
        #     for k in range(len(wind_k)):
        #         wrap_num = wind_k[k]
        #         data_copy = raw_data.copy()
        #         data_copy[:, 1] += 2 * np.pi * wrap_num
        #         t += data_copy * np.tile(r_batch[j,k,:].reshape(-1,1), (1, 2))
        #     S_new[j,:] = np.sum(t, axis=0)
        
        # ### T_k
        # T_new = np.zeros((cluster_nums, 2, 2), dtype=float)
        # for j in range(cluster_nums):
        #     t = np.zeros((num_observations, 2, 2), dtype=float)
        #     for k in range(len(wind_k)):
        #         wrap_num = wind_k[k]
        #         data_copy = raw_data.copy()
        #         data_copy[:, 1] += 2 * np.pi * wrap_num
        #         d_mod = data_copy
        #         t[:,0,0] += d_mod[:,0]**2 * r_batch[j,k,:]
        #         t[:,1,1] += d_mod[:,1]**2 * r_batch[j,k,:]
        #         t[:,0,1] += d_mod[:,0] * d_mod[:, 1] * r_batch[j,k,:]
        #         t[:,1,0] = t[:,0,1]
                
        #     T_new[j,:,:] = np.sum(t, axis=0)