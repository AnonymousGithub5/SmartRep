function deposit ( address _from , bytes32 Parameter_0 , uint _amount ) public { require ( Identifier_0 [ Identifier_1 ] == false , stringLiteral_0 ) ; require ( _amount > 0 , stringLiteral_1 ) ; if ( ! Identifier_2 ( _from , _amount ) ) token . transferFrom ( _from , address ( this ) , _amount ) ; Identifier_3 [ Identifier_4 ] = _amount ; Identifier_5 [ Identifier_6 ] = true ; Identifier_7 [ Identifier_8 ] = _from ; emit Identifier_9 ( _from , Identifier_10 , _amount ) ; }