需要更改两处对于不同物体
##        grasp_pose, _, _ = run_pipeline(
            obj=obj_name, 
            task_string="pick up the knife to cit ",#################################### 
            data_dir="data/", 
            iter="", 
##


    
    # 2. 强制切换工作目录到 ShapeGrasp，确保它能正确读取和生成内部的 data/ outputs/ 文件
    os.chdir("/home/zyp/pan1/ShapeGrasp")
    
    os.makedirs("data", exist_ok=True)
    obj_name = "knife"###############################


    切换3d版本
                output_idx=1, 
            mode="3d", 
            threshold=0.2, 





insert_pos = grasp_pos + grasp_dir * 0.01 # 先到达一个稍微靠近物体的点，增加成功率###################