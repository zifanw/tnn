# tnn

### Sample Usage 

    from layer import Inhibitory_Layer as IL
    from layer import Excitatory_Layer as EL
    from layer import LateralInhibiton_Layer as LL
    import firstlayer as firstlayer

    #Layer initialization
    layer1 = firstlayer.FirstLayer(1)
    layer2 = IL(layer_id=1,
                prev_layer=layer1,
                threshold=3,
                receptive_field=12)

    layer3 = EL(input_dim=8,
                output_dim=16,
                layer_id=3,
                prev_layer=layer2,
                threshold=2,  
                initial_weight=1)
    layer4 = LL(layer_id=4,
                prev_layer=layer3,
                threshold=None,
                receptive_field=None)


    #Forward pass of each layer
    x1 = layer1.forward(mnist.square_data[-10:], 12)
    x2 = layer2.forward(x1, mode='Exact')
    x3 = layer3.forward(x2)
    x4 = layer4.forward(data=x3)
    print (x1)
    print (x2)
    print (x3)
    print (x4)

    #Update Weights of a certain layer:
    layer3 = stdp_update_rule(layer3, x4)
