function FunctionDefinition_0 ( bytes32 Parameter_0 ) external { UserDefinedTypeName_0 storage token = tokens [ _tokenID ] ; UserDefinedTypeName_1 storage VariableDeclaration_0 = token . requests [ token . requests . length - 1 ] ; require ( request . MemberAccess_0 == 0 , stringLiteral_0 ) ; require ( now - request . MemberAccess_1 > Identifier_0 , stringLiteral_1 ) ; if ( token . status == TokenStatus . MemberAccess_2 ) token . status = TokenStatus . MemberAccess_3 ; else if ( token . status == TokenStatus . MemberAccess_4 ) token . status = TokenStatus . MemberAccess_5 ; else revert ( stringLiteral_2 ) ; request . parties [ uint ( Party . Requester ) ] . send ( request . MemberAccess_6 ) ; request . MemberAccess_7 = 0 ; request . MemberAccess_8 = true ; emit Identifier_1 ( request . parties [ uint ( Party . Requester ) ] , request . parties [ uint ( Party . Challenger ) ] , _tokenID , token . status , false , false ) ; }