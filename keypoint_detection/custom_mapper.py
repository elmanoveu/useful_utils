def custom_mapper(dataset_dict,A_transform=A_transform):
    dataset_dict = copy.deepcopy(dataset_dict)     
    prev_anno = dataset_dict["annotations"]
    
    bboxes = np.array([obj["bbox"] for obj in prev_anno],     
 dtype=np.float32)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform =A_transform
    anns = [ann['bbox'] + [ann['category_id']] for ann in dataset_dict['annotations']]
    kps=[ann['keypoints'] for ann in dataset_dict['annotations']]
    
    tr_output = transform(image=image, bboxes=anns,keypoints=kps)
    image = tr_output['image']
    height, width, _ =image.shape

    transform_list = [
    T.Resize((800,800)),T.RandomApply(T.RandomRotation((-5,5),expand=True,sample_style="range",center=[[0.5,0.5],[0.5,0.5]]),prob=0.1),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0,     
 1).astype("float32"))
    annos = [
    utils.transform_instance_annotations(obj, transforms, image.shape[:2])
    for obj in dataset_dict.pop("annotations")
    if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] =utils.filter_empty_instances(instances)
    return dataset_dict