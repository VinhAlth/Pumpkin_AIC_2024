    def search_temporal(self, FEATURES_PATH, queryList, k):
        if self.client.collection_exists(collection_name="temp") == True:
            command = [
                'curl', '-X', 'DELETE', 'http://localhost:6333/collections/'+'temp'
            ]

            # Execute the command
            result = subprocess.run(command)
        

        
        struct_id = 0
        insert_points = []
        SEARCH_RESULTS = self.client.query_points(
            collection_name=self.collection_name, 
            query=queryList[0],
            limit=k,
            timeout=600,
            #score_threshold=0.2
        ).points
        
        return_result = []
        index_dict = {}
        frame_dict = {}
        feat_dict = {}
        for item in SEARCH_RESULTS:
            
            for idx, field in enumerate(item):
                if idx == 0:
                    key = str(field[1])
                elif idx == 2:
                    score = str(field[1])
                elif idx == 3:
                    idx_folder = str(field[1]['idx_folder'])
                    video_name = str(field[1]['video_name'])
                    keyframe_id = str(field[1]['frame_name'])

            result = {"key": key,
                    "idx_folder": idx_folder,
                    "video_name":video_name,
                    "keyframe_id": keyframe_id,
                    "score": score
                    }
            return_result.append(result)
        SEARCH_RESULTS = return_result

        for idx, query in enumerate(queryList):
            if idx==0:
                continue
            st = time.time()

            # self.client.create_collection(
            #     collection_name="temp",
            #     vectors_config=VectorParams(size=self.size, 
            #                                 distance=Distance.COSINE,
            #                                 quantization_config=models.BinaryQuantization(
            #                                     binary=models.BinaryQuantizationConfig(
            #                                     always_ram=True,
            #                                     ),
            #                                 )
            #                                 ),
            #     on_disk_payload=True,
            #     shard_number=24
            # )
            
            command = [
                'curl', '-X', 'PUT', 'http://localhost:6333/collections/'+'temp',
                '-H', 'Content-Type: application/json',
                '--data-raw', '{"vectors": {"size": ' + str(self.size) + ', "distance": "Cosine"}, "quantization_config": {"binary": {"always_ram": true}}}'
            ]

            # Execute the command
            result = subprocess.run(command)
            
            print("Create temporal collection time: ", time.time()-st)
            st = time.time()
            

            for result in SEARCH_RESULTS:
                split_name = result['idx_folder']
                video_name = result['video_name'].replace('.mp4','')
                frame = result['keyframe_id']
                if video_name not in frame_dict.keys():
                    frame_dict[str(video_name)] = get_frame_dict(str(video_name))
                start_index = frame_dict[str(video_name)][str(int(frame))]
                
                if str(video_name) not in index_dict.keys():
                    index_dict[str(video_name)] = get_index_dict(video_name)
                index = start_index
                
                while index < len(index_dict[str(video_name)]) and int(index_dict[str(video_name)][int(str(index))])-int(index_dict[str(video_name)][int(str(start_index))])<1001:
                    if self.valid_program==None or self.valid_program[str((video_name,frame))] == True:
                        
                        if str(video_name) not in feat_dict.keys():
                            feat_dict[str(video_name)] = np.load(FEATURES_PATH[int(split_name)] + '/' + video_name + '.npy')[index]
                        feat_reshaped = feat_dict[str(video_name)][0]
                        insert_points.append(
                            PointStruct(id=struct_id, 
                                vector=feat_reshaped, 
                                payload={
                                    "idx_folder": split_name,
                                    "video_name": video_name + ".mp4",
                                    "frame_name": frame,
                                    "full_name": video_name + ".mp4" + "/" + frame
                            })
                        )
                        struct_id+=1
                        index+=1
            
            print("Process result time: ", time.time()-st)
            
            st = time.time()
            
            operation_info = self.client.upsert(
                collection_name = "temp",
                wait = False,
                points = insert_points)

            print("Upsert time: ", time.time()-st)
            
            st = time.time()
            
            SEARCH_RESULTS = self.client.query_points(
                collection_name="temp", 
                query=query,
                limit=k,
                timeout=600,
                #score_threshold=0.2
            ).points

            print("Query temporal time: ", time.time()-st)
            
            return_result = []
            
            st = time.time()
            
            for item in SEARCH_RESULTS:
                for idx, field in enumerate(item):
                    if idx == 0:
                        key = str(field[1])
                    elif idx == 2:
                        score = str(field[1])
                    elif idx == 3:
                        idx_folder = str(field[1]['idx_folder'])
                        video_name = str(field[1]['video_name'])
                        keyframe_id = str(field[1]['frame_name'])

                result = {"key": key,
                        "idx_folder": idx_folder,
                        "video_name":video_name,
                        "keyframe_id": keyframe_id,
                        "score": score
                        }
                
                if len(return_result)==int(k):
                    break
                return_result.append(result)
                
            SEARCH_RESULTS = return_result
        
            command = [
                'curl', '-X', 'DELETE', 'http://localhost:6333/collections/'+'temp'
            ]

            # Execute the command
            result = subprocess.run(command)
            
            print("Delete collection time: ", time.time()-st)
        
        return SEARCH_RESULTS
        
    def search_temporal_revamp(self, queryList, k):
    
        SEARCH_RESULTS = self.client.query_points(
                            collection_name=self.collection_name, 
                            query=queryList[0],
                            limit=k,
                            timeout=600,
                            #score_threshold=0.2
                        ).points

        return_result = []
        
        for item in SEARCH_RESULTS:
            
            for idx, field in enumerate(item):
                if idx == 0:
                    key = str(field[1])
                elif idx == 2:
                    score = str(field[1])
                elif idx == 3:
                    idx_folder = str(field[1]['idx_folder'])
                    video_name = str(field[1]['video_name'])
                    keyframe_id = str(field[1]['frame_name'])

            result = {"key": key,
                    "idx_folder": idx_folder,
                    "video_name":video_name,
                    "keyframe_id": keyframe_id,
                    "score": score
                    }
            return_result.append(result)
            
        SEARCH_RESULTS = return_result
        st = time.time()
        for idx, query in enumerate(queryList):
            if idx==0:
                continue
            
            return_result = []
            FILTER_RESULTS = {}
            
            for result in SEARCH_RESULTS:
                split_name = result['idx_folder']
                video_name = result['video_name'].replace('.mp4','')
                frame = result['keyframe_id']
                start_index = int(frame_to_index(video_name, frame))
                index_dict = get_index_dict(video_name)
                index = start_index+1
                while index < len(index_dict) and int(index_dict[int(str(index))])-int(index_dict[int(str(start_index))])<=1000:
                    if self.valid_program==None or self.valid_program[str((video_name,frame))] == True:
                        FILTER_RESULTS[(str(video_name), str(frame))]=True
                    index+=1
                    
            SEARCH_RESULTS = self.client.query_points(
                collection_name=self.collection_name, 
                query=query,
                limit=self.getCount(),
                timeout=600,
                #score_threshold=0.2
            ).points
            

            return_result = []
            
            for item in SEARCH_RESULTS:
                for idx, field in enumerate(item):
                    if idx == 0:
                        key = str(field[1])
                    elif idx == 2:
                        score = str(field[1])
                    elif idx == 3:
                        idx_folder = str(field[1]['idx_folder'])
                        video_name = str(field[1]['video_name'])
                        keyframe_id = str(field[1]['frame_name'])

                if (str(video_name.replace(".mp4","")), str(keyframe_id)) in FILTER_RESULTS:
                    
                    result = {"key": key,
                            "idx_folder": idx_folder,
                            "video_name":video_name,
                            "keyframe_id": keyframe_id,
                            "score": score
                            }
                    return_result.append(result)
                
                if len(return_result)>=int(k):
                    break

            SEARCH_RESULTS = return_result
        print("Filter_time: ", time.time()-st)
        return SEARCH_RESULTS
    
    def search_rerank(self, MODEL_B_FEATURES_PATH, SEARCH_RESULTS_A, query, size, k):
        
        if self.client.collection_exists(collection_name="temp") == True:
            self.client.delete_collection(collection_name="temp")
        
        self.client.create_collection(
            collection_name="temp",
            vectors_config=VectorParams(size=size, 
                                        distance=Distance.COSINE,
                                        quantization_config=models.BinaryQuantization(
                                            binary=models.BinaryQuantizationConfig(
                                            always_ram=True,
                                            ),
                                        )
                                        ),
            on_disk_payload=True,
            shard_number=96
        )

        struct_id = 0
        insert_points = []
        
        for result in SEARCH_RESULTS_A:
            split_name = result['idx_folder']
            video_name = result['video_name'].replace('.mp4','')
            frame = result['keyframe_id'].zfill(5)
            frame_index = frame_to_index(video_name, frame)
            feat = np.load(MODEL_B_FEATURES_PATH[int(split_name)] + '/' + video_name + '.npy')[frame_index]
            feat_reshaped = feat.reshape(1,-1).astype('float32')[0]
            insert_points.append(
                PointStruct(id=struct_id, 
                    vector=feat_reshaped, 
                    payload={
                        "idx_folder": split_name,
                        "video_name": video_name + ".mp4",
                        "frame_name": int(frame)
                })
            )
            struct_id+=1
            
        operation_info = self.client.upsert(
            collection_name = "temp",
            wait = False,
            points = insert_points)
        
        search_results = self.client.query_points(
            collection_name="temp", 
            query=query,
            limit=k,
            timeout=600,
            #score_threshold=0.2
        ).points
        
        return_result = []
        
        for item in search_results:
            
            for idx, field in enumerate(item):
                if idx == 0:
                    key = str(field[1])
                elif idx == 2:
                    score = str(field[1])
                elif idx == 3:
                    idx_folder = str(field[1]['idx_folder'])
                    video_name = str(field[1]['video_name'])
                    keyframe_id = str(field[1]['frame_name']).zfill(5)

            result = {"key": key,
                      "idx_folder": idx_folder,
                      "video_name":video_name,
                      "keyframe_id": keyframe_id,
                      "score": score
                      }
            return_result.append(result)
        
        self.client.delete_collection(collection_name="temp")
        
        return return_result
    
    