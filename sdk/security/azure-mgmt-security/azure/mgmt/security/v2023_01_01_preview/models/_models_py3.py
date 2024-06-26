# coding=utf-8
# pylint: disable=too-many-lines
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from typing import Any, List, Literal, Optional, TYPE_CHECKING

from ... import _serialization

if TYPE_CHECKING:
    # pylint: disable=unused-import,ungrouped-imports
    from .. import models as _models


class CloudErrorBody(_serialization.Model):
    """The error detail.

    Variables are only populated by the server, and will be ignored when sending a request.

    :ivar code: The error code.
    :vartype code: str
    :ivar message: The error message.
    :vartype message: str
    :ivar target: The error target.
    :vartype target: str
    :ivar details: The error details.
    :vartype details: list[~azure.mgmt.security.v2023_01_01_preview.models.CloudErrorBody]
    :ivar additional_info: The error additional info.
    :vartype additional_info:
     list[~azure.mgmt.security.v2023_01_01_preview.models.ErrorAdditionalInfo]
    """

    _validation = {
        "code": {"readonly": True},
        "message": {"readonly": True},
        "target": {"readonly": True},
        "details": {"readonly": True},
        "additional_info": {"readonly": True},
    }

    _attribute_map = {
        "code": {"key": "code", "type": "str"},
        "message": {"key": "message", "type": "str"},
        "target": {"key": "target", "type": "str"},
        "details": {"key": "details", "type": "[CloudErrorBody]"},
        "additional_info": {"key": "additionalInfo", "type": "[ErrorAdditionalInfo]"},
    }

    def __init__(self, **kwargs: Any) -> None:
        """ """
        super().__init__(**kwargs)
        self.code = None
        self.message = None
        self.target = None
        self.details = None
        self.additional_info = None


class ErrorAdditionalInfo(_serialization.Model):
    """The resource management error additional info.

    Variables are only populated by the server, and will be ignored when sending a request.

    :ivar type: The additional info type.
    :vartype type: str
    :ivar info: The additional info.
    :vartype info: JSON
    """

    _validation = {
        "type": {"readonly": True},
        "info": {"readonly": True},
    }

    _attribute_map = {
        "type": {"key": "type", "type": "str"},
        "info": {"key": "info", "type": "object"},
    }

    def __init__(self, **kwargs: Any) -> None:
        """ """
        super().__init__(**kwargs)
        self.type = None
        self.info = None


class Identity(_serialization.Model):
    """Identity for the resource.

    Variables are only populated by the server, and will be ignored when sending a request.

    :ivar principal_id: The principal ID of resource identity.
    :vartype principal_id: str
    :ivar tenant_id: The tenant ID of resource.
    :vartype tenant_id: str
    :ivar type: The identity type. Default value is "SystemAssigned".
    :vartype type: str
    """

    _validation = {
        "principal_id": {"readonly": True},
        "tenant_id": {"readonly": True},
    }

    _attribute_map = {
        "principal_id": {"key": "principalId", "type": "str"},
        "tenant_id": {"key": "tenantId", "type": "str"},
        "type": {"key": "type", "type": "str"},
    }

    def __init__(self, *, type: Optional[Literal["SystemAssigned"]] = None, **kwargs: Any) -> None:
        """
        :keyword type: The identity type. Default value is "SystemAssigned".
        :paramtype type: str
        """
        super().__init__(**kwargs)
        self.principal_id = None
        self.tenant_id = None
        self.type = type


class Resource(_serialization.Model):
    """Describes an Azure resource.

    Variables are only populated by the server, and will be ignored when sending a request.

    :ivar id: Resource Id.
    :vartype id: str
    :ivar name: Resource name.
    :vartype name: str
    :ivar type: Resource type.
    :vartype type: str
    """

    _validation = {
        "id": {"readonly": True},
        "name": {"readonly": True},
        "type": {"readonly": True},
    }

    _attribute_map = {
        "id": {"key": "id", "type": "str"},
        "name": {"key": "name", "type": "str"},
        "type": {"key": "type", "type": "str"},
    }

    def __init__(self, **kwargs: Any) -> None:
        """ """
        super().__init__(**kwargs)
        self.id = None
        self.name = None
        self.type = None


class SecurityOperator(Resource):
    """Security operator under a given subscription and pricing.

    Variables are only populated by the server, and will be ignored when sending a request.

    :ivar id: Resource Id.
    :vartype id: str
    :ivar name: Resource name.
    :vartype name: str
    :ivar type: Resource type.
    :vartype type: str
    :ivar identity: Identity for the resource.
    :vartype identity: ~azure.mgmt.security.v2023_01_01_preview.models.Identity
    """

    _validation = {
        "id": {"readonly": True},
        "name": {"readonly": True},
        "type": {"readonly": True},
    }

    _attribute_map = {
        "id": {"key": "id", "type": "str"},
        "name": {"key": "name", "type": "str"},
        "type": {"key": "type", "type": "str"},
        "identity": {"key": "identity", "type": "Identity"},
    }

    def __init__(self, *, identity: Optional["_models.Identity"] = None, **kwargs: Any) -> None:
        """
        :keyword identity: Identity for the resource.
        :paramtype identity: ~azure.mgmt.security.v2023_01_01_preview.models.Identity
        """
        super().__init__(**kwargs)
        self.identity = identity


class SecurityOperatorList(_serialization.Model):
    """List of SecurityOperator response.

    All required parameters must be populated in order to send to server.

    :ivar value: List of SecurityOperator configurations. Required.
    :vartype value: list[~azure.mgmt.security.v2023_01_01_preview.models.SecurityOperator]
    """

    _validation = {
        "value": {"required": True},
    }

    _attribute_map = {
        "value": {"key": "value", "type": "[SecurityOperator]"},
    }

    def __init__(self, *, value: List["_models.SecurityOperator"], **kwargs: Any) -> None:
        """
        :keyword value: List of SecurityOperator configurations. Required.
        :paramtype value: list[~azure.mgmt.security.v2023_01_01_preview.models.SecurityOperator]
        """
        super().__init__(**kwargs)
        self.value = value
