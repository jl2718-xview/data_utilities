import object_detection.protos.string_int_label_map_pb2 as proto


A = proto.StringIntLabelMap()
with open('./xview_class_labels.txt','r')  as f:
    for s in f:
        id,desc = s.strip().split(':')
        a = A.item.add()
        a.id = int(id)
        a.display_name = desc

print(A)