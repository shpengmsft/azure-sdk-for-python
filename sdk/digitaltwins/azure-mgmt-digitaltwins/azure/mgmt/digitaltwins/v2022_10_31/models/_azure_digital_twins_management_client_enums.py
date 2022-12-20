# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from enum import Enum
from azure.core import CaseInsensitiveEnumMeta


class AuthenticationType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Specifies the authentication type being used for connecting to the endpoint. Defaults to
    'KeyBased'. If 'KeyBased' is selected, a connection string must be specified (at least the
    primary connection string). If 'IdentityBased' is select, the endpointUri and entityPath
    properties must be specified.
    """

    KEY_BASED = "KeyBased"
    IDENTITY_BASED = "IdentityBased"


class ConnectionPropertiesProvisioningState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The provisioning state."""

    PENDING = "Pending"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    DISCONNECTED = "Disconnected"


class ConnectionType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The type of time series connection resource."""

    AZURE_DATA_EXPLORER = "AzureDataExplorer"


class CreatedByType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The type of identity that created the resource."""

    USER = "User"
    APPLICATION = "Application"
    MANAGED_IDENTITY = "ManagedIdentity"
    KEY = "Key"


class DigitalTwinsIdentityType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The type of Managed Identity used by the DigitalTwinsInstance."""

    NONE = "None"
    SYSTEM_ASSIGNED = "SystemAssigned"
    USER_ASSIGNED = "UserAssigned"
    SYSTEM_ASSIGNED_USER_ASSIGNED = "SystemAssigned,UserAssigned"


class EndpointProvisioningState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The provisioning state."""

    PROVISIONING = "Provisioning"
    DELETING = "Deleting"
    UPDATING = "Updating"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELED = "Canceled"
    DELETED = "Deleted"
    WARNING = "Warning"
    SUSPENDING = "Suspending"
    RESTORING = "Restoring"
    MOVING = "Moving"
    DISABLED = "Disabled"


class EndpointType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The type of Digital Twins endpoint."""

    EVENT_HUB = "EventHub"
    EVENT_GRID = "EventGrid"
    SERVICE_BUS = "ServiceBus"


class IdentityType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The type of managed identity used."""

    SYSTEM_ASSIGNED = "SystemAssigned"
    USER_ASSIGNED = "UserAssigned"


class PrivateLinkServiceConnectionStatus(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The status of a private endpoint connection."""

    PENDING = "Pending"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    DISCONNECTED = "Disconnected"


class ProvisioningState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The provisioning state."""

    PROVISIONING = "Provisioning"
    DELETING = "Deleting"
    UPDATING = "Updating"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELED = "Canceled"
    DELETED = "Deleted"
    WARNING = "Warning"
    SUSPENDING = "Suspending"
    RESTORING = "Restoring"
    MOVING = "Moving"


class PublicNetworkAccess(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Public network access for the DigitalTwinsInstance."""

    ENABLED = "Enabled"
    DISABLED = "Disabled"


class Reason(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Message providing the reason why the given name is invalid."""

    INVALID = "Invalid"
    ALREADY_EXISTS = "AlreadyExists"


class TimeSeriesDatabaseConnectionState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The provisioning state."""

    PROVISIONING = "Provisioning"
    DELETING = "Deleting"
    UPDATING = "Updating"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELED = "Canceled"
    DELETED = "Deleted"
    WARNING = "Warning"
    SUSPENDING = "Suspending"
    RESTORING = "Restoring"
    MOVING = "Moving"
    DISABLED = "Disabled"